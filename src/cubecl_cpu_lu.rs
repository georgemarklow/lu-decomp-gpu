use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;

/// Computes the LU factorization of an n x n column-major matrix in place
/// using CubeCL's CPU runtime, with partial pivoting.
///
/// `threads` is the number of CPU threads used for the trailing update.
/// In the CubeCL CPU runtime each unit of a cube runs on its own worker
/// thread, so this is passed as the cube size (CubeDim).
///
/// The kernels avoid nested loops, loops inside `if` blocks, and repeated
/// loops in one kernel, which the CubeCL 0.7 CPU compiler (MLIR) cannot
/// compile. To avoid an `if j < n` bounds check, the device buffer has
/// `threads` spare columns of padding that out-of-range threads write to.
///
/// Output layout matches LAPACK sgetrf: U in the upper triangle, L (whose
/// diagonal of 1s is not stored) in the strict lower triangle, and 1-based
/// pivot indices.
#[allow(dead_code)]
pub fn run_cubecl_cpu_lu(matrix: &mut [f32], n: usize, threads: u32) -> Vec<i32> {
    assert_eq!(matrix.len(), n * n, "matrix must be n x n");
    assert!(threads > 0, "threads must be at least 1");

    let client = CpuRuntime::client(&CpuDevice::default());
    let n_u32 = n as u32;

    // Matrix followed by `threads` zero columns of padding.
    let padded_len = n * (n + threads as usize);
    let mut padded = vec![0.0f32; padded_len];
    padded[..n * n].copy_from_slice(matrix);

    let a = client.create(f32::as_bytes(&padded));
    let piv = client.empty(n * std::mem::size_of::<u32>());
    let scratch = client.empty(2 * n * std::mem::size_of::<f32>()); // rows k and p

    for k in 0..n_u32 {
        let m = n_u32 - k - 1; // number of trailing rows and columns

        unsafe {
            // 1. Find the pivot row for column k (one thread).
            find_pivot::launch_unchecked::<f32, CpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a, padded_len, 1),
                ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                ScalarArg::new(n_u32),
                ScalarArg::new(k),
            );

            // 2. Swap rows k and p in every column, via the scratch buffer
            //    (one thread, one cube per column).
            copy_rows_out::launch_unchecked::<f32, CpuRuntime>(
                &client,
                CubeCount::Static(n_u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a, padded_len, 1),
                ArrayArg::from_raw_parts::<f32>(&scratch, 2 * n, 1),
                ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                ScalarArg::new(n_u32),
                ScalarArg::new(k),
            );
            copy_rows_back::launch_unchecked::<f32, CpuRuntime>(
                &client,
                CubeCount::Static(n_u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a, padded_len, 1),
                ArrayArg::from_raw_parts::<f32>(&scratch, 2 * n, 1),
                ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                ScalarArg::new(n_u32),
                ScalarArg::new(k),
            );

            if m > 0 {
                // 3. Divide column k below the diagonal by the pivot
                //    (one thread, one cube per row).
                scale_column::launch_unchecked::<f32, CpuRuntime>(
                    &client,
                    CubeCount::Static(m, 1, 1),
                    CubeDim::new(1, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, padded_len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k),
                );

                // 4. Trailing update on `threads` threads, one column per unit.
                update_column::launch_unchecked::<f32, CpuRuntime>(
                    &client,
                    CubeCount::Static(m.div_ceil(threads), 1, 1),
                    CubeDim::new(threads, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, padded_len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k),
                );
            }
        }
    }

    let a_bytes = client.read_one(a);
    matrix.copy_from_slice(&f32::from_bytes(&a_bytes)[..n * n]);

    let piv_bytes = client.read_one(piv);
    u32::from_bytes(&piv_bytes)
        .iter()
        .map(|&p| p as i32 + 1) // 0-based -> 1-based, like LAPACK
        .collect()
}

/// Single thread: finds the row with the largest |A[i, k]| for i >= k.
/// Ties go to the lowest row index, matching LAPACK's isamax.
/// Loops over an offset from 0 rather than over k+1..n: the CubeCL 0.7
/// CPU compiler rejects the latter form of this loop.
#[cube(launch_unchecked)]
pub fn find_pivot<F: Float>(a: &Array<F>, piv: &mut Array<u32>, n: u32, k: u32) {
    let base = k * n + k; // A(k, k)
    let count = n - k;
    let mut best = 0;
    let mut best_val = F::abs(a[base]);
    for t in 1..count {
        let v = F::abs(a[base + t]);
        if v > best_val {
            best_val = v;
            best = t;
        }
    }
    piv[k] = k + best;
}

/// One unit per column j: copies A(k, j) and A(p, j) into the scratch buffer.
#[cube(launch_unchecked)]
pub fn copy_rows_out<F: Float>(a: &Array<F>, scratch: &mut Array<F>, piv: &Array<u32>, n: u32, k: u32) {
    let j = ABSOLUTE_POS;
    let p = piv[k];
    scratch[j] = a[j * n + k];
    scratch[n + j] = a[j * n + p];
}

/// One unit per column j: writes the saved rows back swapped. When p == k
/// both writes store the original value.
///
/// The swap needs two kernels because the CubeCL 0.7 CPU backend defers
/// loads until a value is used: in a single-kernel swap the saved value is
/// read after it has already been overwritten.
#[cube(launch_unchecked)]
pub fn copy_rows_back<F: Float>(a: &mut Array<F>, scratch: &Array<F>, piv: &Array<u32>, n: u32, k: u32) {
    let j = ABSOLUTE_POS;
    let p = piv[k];
    a[j * n + p] = scratch[j];
    a[j * n + k] = scratch[n + j];
}

/// One unit per row i > k: A(i, k) /= A(k, k).
#[cube(launch_unchecked)]
pub fn scale_column<F: Float>(a: &mut Array<F>, n: u32, k: u32) {
    let i = k + 1 + ABSOLUTE_POS;
    a[k * n + i] = a[k * n + i] / a[k * n + k];
}

/// One unit per column j > k: A(i, j) -= L(i, k) * U(k, j) for all i > k.
/// Units past the last column work on the padding columns.
#[cube(launch_unchecked)]
pub fn update_column<F: Float>(a: &mut Array<F>, n: u32, k: u32) {
    let j = k + 1 + ABSOLUTE_POS;
    let u = a[j * n + k]; // U(k, j)
    for i in k + 1..n {
        a[j * n + i] = a[j * n + i] - a[k * n + i] * u;
    }
}
