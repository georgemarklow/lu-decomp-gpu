use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::prelude::*;

/// Computes the LU factorization of an n x n column-major matrix in place
/// using CubeCL's CPU runtime, with partial pivoting.
///
/// Output layout matches LAPACK sgetrf: U in the upper triangle, L (whose
/// diagonal of 1s is not stored) in the strict lower triangle, and 1-based
/// pivot indices.
pub fn run_cubecl_cpu_lu(matrix: &mut [f32], n: usize) -> Vec<i32> {
    assert_eq!(matrix.len(), n * n, "matrix must be n x n");

    let client = CpuRuntime::client(&CpuDevice::default());
    let len = n * n;
    let n_u32 = n as u32;

    let a = client.create(f32::as_bytes(matrix));
    let piv = client.empty(n * std::mem::size_of::<u32>());

    for k in 0..n {
        let k_u32 = k as u32;
        let m = n_u32 - k_u32 - 1; // number of trailing columns

        unsafe {
            // 1. Pivot search, row swap and column scaling, all in one
            //    serial kernel. Each step is O(n), which a CPU core handles
            //    faster than a launch across many threads.
            pivot_swap_scale::launch_unchecked::<f32, CpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                ArrayArg::from_raw_parts::<u32>(&piv, n, 1),
                ScalarArg::new(n_u32),
                ScalarArg::new(k_u32),
            );

            // 2. Trailing update: one thread per column, each looping down
            //    its column, so every thread walks contiguous memory.
            if m > 0 {
                update_columns::launch_unchecked::<f32, CpuRuntime>(
                    &client,
                    CubeCount::Static(m.div_ceil(CPU_THREADS), 1, 1),
                    CubeDim::new(CPU_THREADS, 1, 1),
                    ArrayArg::from_raw_parts::<f32>(&a, len, 1),
                    ScalarArg::new(n_u32),
                    ScalarArg::new(k_u32),
                );
            }
        }
    }

    let a_bytes = client.read_one(a);
    matrix.copy_from_slice(f32::from_bytes(&a_bytes));

    let piv_bytes = client.read_one(piv);
    u32::from_bytes(&piv_bytes)
        .iter()
        .map(|&p| p as i32 + 1) // 0-based -> 1-based, like LAPACK
        .collect()
}

const CPU_THREADS: u32 = 64;

/// Single thread: finds the pivot in column k, swaps rows k and p across
/// all n columns, then divides the entries below the pivot by the pivot.
/// Ties go to the lowest row index, matching LAPACK's isamax.
#[cube(launch_unchecked)]
fn pivot_swap_scale<F: Float>(a: &mut Array<F>, piv: &mut Array<u32>, n: u32, k: u32) {
    if ABSOLUTE_POS == 0 {
        // Find the row with the largest |A[i, k]| for i >= k.
        let mut best = k;
        let mut best_val = F::abs(a[k * n + k]);
        for i in k + 1..n {
            let v = F::abs(a[k * n + i]);
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        piv[k] = best;

        // Swap rows k and best across all columns.
        if best != k {
            for j in 0..n {
                let tmp = a[j * n + k];
                a[j * n + k] = a[j * n + best];
                a[j * n + best] = tmp;
            }
        }

        // Form column k of L.
        let pivot = a[k * n + k];
        if pivot != F::new(0.0_f32) {
            for i in k + 1..n {
                a[k * n + i] = a[k * n + i] / pivot;
            }
        }
    }
}

/// One thread per trailing column j > k:
/// A[i, j] -= L(i, k) * U(k, j) for all i > k.
#[cube(launch_unchecked)]
fn update_columns<F: Float>(a: &mut Array<F>, n: u32, k: u32) {
    let j = k + 1 + ABSOLUTE_POS;
    if j < n {
        let u = a[j * n + k]; // U(k, j), loaded once per column
        for i in k + 1..n {
            a[j * n + i] = a[j * n + i] - a[k * n + i] * u;
        }
    }
}