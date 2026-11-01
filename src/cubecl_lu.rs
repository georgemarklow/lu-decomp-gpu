use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

/// Computes the LU factorization of an n x n column-major matrix in place
/// on the GPU, using CubeCL with partial pivoting.
///
/// Output layout matches LAPACK sgetrf: U in the upper triangle, L (whose
/// diagonal of 1s is not stored) in the strict lower triangle, and 1-based
/// pivot indices.
pub fn run_cubecl_gpu_lu(matrix: &mut [f32], n: usize) -> Vec<i32> {
    run_lu::<WgpuRuntime>(&WgpuDevice::default(), matrix, n)
}

fn run_lu<R: Runtime>(device: &R::Device, matrix: &mut [f32], n: usize) -> Vec<i32> {
    assert_eq!(matrix.len(), n * n, "matrix must be n x n");

    let client = R::client(device);
    let len = n * n;
    let n_u32 = n as u32;

    // Copy the matrix to the GPU and allocate the pivot array.
    let a = client.create(f32::as_bytes(matrix));
    let piv = client.empty(n * std::mem::size_of::<u32>());

    for k in 0..n {
        let k_u32 = k as u32;
        let m = n_u32 - k_u32 - 1; // size of the trailing submatrix

        unsafe {
            // 1. Find the row with the largest |A[i, k]| for i >= k.
            find_pivot::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(PIVOT_THREADS, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                ScalarArg::new(n_u32),
                ScalarArg::new(k_u32),
            );

            // 2. Swap row k with the pivot row, across all n columns.
            swap_rows::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(n_u32.div_ceil(256), 1, 1),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                ScalarArg::new(n_u32),
                ScalarArg::new(k_u32),
            );

            if m > 0 {
                // 3. Divide the column below the pivot by the pivot (forms L).
                scale_column::launch_unchecked::<f32, R>(
                    &client,
                    CubeCount::Static(m.div_ceil(256), 1, 1),
                    CubeDim::new(256, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k_u32),
                );

                // 4. Rank-1 update of the trailing submatrix.
                update_trailing::launch_unchecked::<f32, R>(
                    &client,
                    CubeCount::Static(m.div_ceil(16), m.div_ceil(16), 1),
                    CubeDim::new(16, 16, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k_u32),
                );
            }
        }
    }

    // Copy the factored matrix and the pivots back to the CPU.
    let a_bytes = client.read_one(a.binding());
    matrix.copy_from_slice(f32::from_bytes(&a_bytes));

    let piv_bytes = client.read_one(piv.binding());
    u32::from_bytes(&piv_bytes)
        .iter()
        .map(|&p| p as i32 + 1) // 0-based -> 1-based, like LAPACK
        .collect()
}

const PIVOT_THREADS: u32 = 256;

/// One cube of PIVOT_THREADS threads: finds the index of the largest
/// |A[i, k]| for i >= k using a shared-memory reduction.
/// Ties go to the lowest row index, matching LAPACK's isamax.
#[cube(launch_unchecked)]
fn find_pivot<F: Float>(a: &Array<F>, piv: &mut Array<u32>, n: u32, k: u32) {
    let tid = UNIT_POS;
    let mut vals = SharedMemory::<F>::new(256);
    let mut idxs = SharedMemory::<u32>::new(256);

    // Each thread scans rows k + tid, k + tid + CUBE_DIM_X, ...
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

    // Tree reduction: 256 -> 128 -> ... -> 1.
    // stride starts from CUBE_DIM_X so it is a runtime variable, not a constant.
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
        piv[k] = idxs[0];
    }
}

/// One thread per column j: swaps A[k, j] and A[p, j].
#[cube(launch_unchecked)]
fn swap_rows<F: Float>(a: &mut Array<F>, piv: &Array<u32>, n: u32, k: u32) {
    let j = ABSOLUTE_POS;
    if j < n {
        let p = piv[k];
        if p != k {
            let tmp = a[j * n + k];
            a[j * n + k] = a[j * n + p];
            a[j * n + p] = tmp;
        }
    }
}

/// One thread per row i > k: A[i, k] /= A[k, k].
#[cube(launch_unchecked)]
fn scale_column<F: Float>(a: &mut Array<F>, n: u32, k: u32) {
    let i = k + 1 + ABSOLUTE_POS;
    if i < n {
        let pivot = a[k * n + k];
        if pivot != F::new(0.0) {
            a[k * n + i] = a[k * n + i] / pivot;
        }
    }
}

/// One thread per (i, j) with i, j > k: A[i, j] -= A[i, k] * A[k, j].
#[cube(launch_unchecked)]
fn update_trailing<F: Float>(a: &mut Array<F>, n: u32, k: u32) {
    let i = k + 1 + ABSOLUTE_POS_X;
    let j = k + 1 + ABSOLUTE_POS_Y;
    if i < n && j < n {
        a[j * n + i] = a[j * n + i] - a[k * n + i] * a[j * n + k];
    }
}