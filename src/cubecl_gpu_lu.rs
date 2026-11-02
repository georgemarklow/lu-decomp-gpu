use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

/// Panel width for the blocked algorithm. Try 32, 64 or 128.
const NB: u32 = 64;

/// Tile size for the GEMM update (GEMM_TILE x GEMM_TILE threads per cube).
const GEMM_TILE: u32 = 16;

/// Computes the LU factorization of an n x n column-major matrix in place
/// on the GPU, using a blocked right-looking algorithm with partial pivoting.
///
/// `threads` is the number of threads per cube for the 1D kernels. It must
/// be a power of two.
///
/// Output layout matches LAPACK sgetrf: U in the upper triangle, L (whose
/// diagonal of 1s is not stored) in the strict lower triangle, and 1-based
/// pivot indices.
pub fn run_cubecl_gpu_lu(matrix: &mut [f32], n: usize, threads: u32) -> Vec<i32> {
    run_lu::<WgpuRuntime>(&WgpuDevice::default(), matrix, n, threads)
}

fn run_lu<R: Runtime>(device: &R::Device, matrix: &mut [f32], n: usize, threads: u32) -> Vec<i32> {
    assert_eq!(matrix.len(), n * n, "matrix must be n x n");
    assert!(threads.is_power_of_two(), "threads must be a power of two");

    let client = R::client(device);
    let len = n * n;
    let n_u32 = n as u32;
    let tile = threads.isqrt();

    // Copy the matrix to the GPU and allocate the pivot array.
    let a = client.create(f32::as_bytes(matrix));
    let piv = client.empty(n * std::mem::size_of::<u32>());

    let mut k0 = 0u32;
    while k0 < n_u32 {
        let b = Ord::min(NB, n_u32 - k0);
        let k1 = k0 + b; // panel is columns k0..k1

        unsafe {
            // 1-2. Factor the panel one column at a time.
            for c in k0..k1 {
                pivot_swap_scale::launch_unchecked::<f32, R>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new(threads, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(c),
                    ScalarArg::new(k0),
                    ScalarArg::new(k1),
                    threads,
                );

                let rows = n_u32 - c - 1;
                let cols = k1 - c - 1;
                if rows > 0 && cols > 0 {
                    update_panel::launch_unchecked::<f32, R>(
                        &client,
                        CubeCount::Static(rows.div_ceil(tile), cols.div_ceil(tile), 1),
                        CubeDim::new(tile, tile, 1),
                        ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                        ScalarArg::new(n_u32),
                        ScalarArg::new(c),
                        ScalarArg::new(k1),
                    );
                }
            }

            // 3. Apply this panel's row swaps to every column outside it.
            if b < n_u32 {
                apply_swaps::launch_unchecked::<f32, R>(
                    &client,
                    CubeCount::Static(n_u32.div_ceil(threads), 1, 1),
                    CubeDim::new(threads, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k0),
                    ScalarArg::new(k1),
                );
            }

            let rest = n_u32 - k1;
            if rest > 0 {
                // 4. U12 = L11^-1 * A12 (forward substitution, one thread per column).
                trsm_panel::launch_unchecked::<f32, R>(
                    &client,
                    CubeCount::Static(rest.div_ceil(threads), 1, 1),
                    CubeDim::new(threads, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k0),
                    ScalarArg::new(k1),
                );

                // 5. A22 -= L21 * U12 (tiled matrix multiply).
                gemm_update::launch_unchecked::<f32, R>(
                    &client,
                    CubeCount::Static(rest.div_ceil(GEMM_TILE), rest.div_ceil(GEMM_TILE), 1),
                    CubeDim::new(GEMM_TILE, GEMM_TILE, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k0),
                    ScalarArg::new(k1),
                    GEMM_TILE,
                );
            }
        }

        k0 = k1;
    }

    // Copy the factored matrix and the pivots back to the CPU.
    let a_bytes = client.read_one(a);
    matrix.copy_from_slice(f32::from_bytes(&a_bytes));

    let piv_bytes = client.read_one(piv);
    u32::from_bytes(&piv_bytes)
        .iter()
        .map(|&p| p as i32 + 1) // 0-based -> 1-based, like LAPACK
        .collect()
}

/// One cube: finds the pivot in column k, swaps rows k and p in the panel
/// columns col_start..col_end, and divides column k below the diagonal by
/// the pivot. Ties go to the lowest row index, matching LAPACK's isamax.
///
/// Column k is swapped and scaled in a single pass, so no thread reads a
/// global value another thread wrote. Barriers only cover shared memory.
#[cube(launch_unchecked)]
fn pivot_swap_scale<F: Float>(
    a: &mut Array<F>,
    piv: &mut Array<u32>,
    n: u32,
    k: u32,
    col_start: u32,
    col_end: u32,
    #[comptime] threads: u32,
) {
    let tid = UNIT_POS;
    let mut vals = SharedMemory::<F>::new(threads);
    let mut idxs = SharedMemory::<u32>::new(threads);
    let mut scalars = SharedMemory::<F>::new(2); // [pivot value, old A(k, k)]

    // Phase 1: each thread scans rows k + tid, k + tid + CUBE_DIM_X, ...
    let mut best = k;
    let mut best_val = F::new(-1.0_f32);
    let mut i = k + tid;
    while i < n {
        let v = F::abs(a[k * n + i]);
        if v > best_val {
            best_val = v;
            best = i;
        }
        i += CUBE_DIM_X;
    }
    vals[tid] = best_val;
    idxs[tid] = best;
    sync_cube();

    // Tree reduction: threads -> threads/2 -> ... -> 1.
    let mut stride = CUBE_DIM_X / 2;
    while stride > 0 {
        if tid < stride {
            let other_val = vals[tid + stride];
            let other_idx = idxs[tid + stride];
            if other_val > vals[tid] || (other_val == vals[tid] && other_idx < idxs[tid]) {
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }
        }
        sync_cube();
        stride /= 2;
    }

    if tid == 0 {
        let p0 = idxs[0];
        piv[k] = p0;
        scalars[0] = a[k * n + p0];
        scalars[1] = a[k * n + k];
    }
    sync_cube();

    let p = idxs[0];
    let pv = scalars[0];
    let old_kk = scalars[1];

    // Phase 2a: swap rows k and p in the panel columns other than k.
    if p != k {
        let mut j = col_start + tid;
        while j < col_end {
            if j != k {
                let tmp = a[j * n + k];
                a[j * n + k] = a[j * n + p];
                a[j * n + p] = tmp;
            }
            j += CUBE_DIM_X;
        }
    }

    // Phase 2b: column k, swapped and scaled in one pass.
    let mut r = k + tid;
    while r < n {
        if r == k {
            a[k * n + k] = pv;
        } else {
            let mut v = a[k * n + r];
            if r == p {
                v = old_kk;
            }
            if pv != F::new(0.0_f32) {
                v = v / pv;
            }
            a[k * n + r] = v;
        }
        r += CUBE_DIM_X;
    }
}

/// Rank-1 update inside the panel: for i > k and k < j < col_end,
/// A[i, j] -= A[i, k] * A[k, j].
#[cube(launch_unchecked)]
fn update_panel<F: Float>(a: &mut Array<F>, n: u32, k: u32, col_end: u32) {
    let i = k + 1 + ABSOLUTE_POS_X;
    let j = k + 1 + ABSOLUTE_POS_Y;
    if i < n && j < col_end {
        a[j * n + i] = a[j * n + i] - a[k * n + i] * a[j * n + k];
    }
}

/// One thread per column j outside the panel k0..k1: applies the panel's
/// row swaps in order.
#[cube(launch_unchecked)]
fn apply_swaps<F: Float>(a: &mut Array<F>, piv: &Array<u32>, n: u32, k0: u32, k1: u32) {
    let j = ABSOLUTE_POS;
    if j < n && (j < k0 || j >= k1) {
        for c in k0..k1 {
            let p = piv[c];
            if p != c {
                let tmp = a[j * n + c];
                a[j * n + c] = a[j * n + p];
                a[j * n + p] = tmp;
            }
        }
    }
}

/// One thread per column j >= k1: solves L11 * U12 = A12 in place, where
/// L11 is the unit lower triangle of the panel's diagonal block.
#[cube(launch_unchecked)]
fn trsm_panel<F: Float>(a: &mut Array<F>, n: u32, k0: u32, k1: u32) {
    let j = k1 + ABSOLUTE_POS;
    if j < n {
        for r in k0..k1 {
            let x = a[j * n + r];
            for i in r + 1..k1 {
                a[j * n + i] = a[j * n + i] - a[r * n + i] * x;
            }
        }
    }
}

/// Tiled update of the trailing matrix: A22 -= L21 * U12, for rows and
/// columns >= k1, summing over the panel columns k0..k1. Each cube computes
/// one tile x tile block of A22, staging L21 and U12 through shared memory.
#[cube(launch_unchecked)]
fn gemm_update<F: Float>(a: &mut Array<F>, n: u32, k0: u32, k1: u32, #[comptime] tile: u32) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let i = k1 + CUBE_POS_X * tile + tx; // row in A22
    let j = k1 + CUBE_POS_Y * tile + ty; // column in A22

    let mut l_tile = SharedMemory::<F>::new(tile * tile); // [c][i]
    let mut u_tile = SharedMemory::<F>::new(tile * tile); // [j][c]

    let mut acc = F::new(0.0_f32);
    let mut c0 = k0;
    while c0 < k1 {
        // Load L21(i, c0 + ty) and U12(c0 + tx, j). Neighbouring threads
        // (tx) read neighbouring rows, so both loads are coalesced.
        let lc = c0 + ty;
        if i < n && lc < k1 {
            l_tile[ty * tile + tx] = a[lc * n + i];
        } else {
            l_tile[ty * tile + tx] = F::new(0.0_f32);
        }

        let ur = c0 + tx;
        if j < n && ur < k1 {
            u_tile[ty * tile + tx] = a[j * n + ur];
        } else {
            u_tile[ty * tile + tx] = F::new(0.0_f32);
        }
        sync_cube();

        for t in 0..tile {
            acc += l_tile[t * tile + tx] * u_tile[ty * tile + t];
        }
        sync_cube();

        c0 += tile;
    }

    if i < n && j < n {
        a[j * n + i] = a[j * n + i] - acc;
    }
}