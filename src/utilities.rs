/// Relative residual ||P*A - L*U||_F / ||A||_F of a factorization produced
/// by run_lapack_lu or run_cubecl_gpu_lu. Accumulated in f64 so the check
/// itself doesn't add f32 rounding error.
pub fn lu_residual(original: &[f32], lu: &[f32], ipiv: &[i32], n: usize) -> f64 {
    // P * A: apply the row swaps to a copy of the original, in order.
    let mut pa: Vec<f64> = original.iter().map(|&x| x as f64).collect();
    for k in 0..n {
        let p = (ipiv[k] - 1) as usize; // pivots are 1-based
        if p != k {
            for j in 0..n {
                pa.swap(j * n + k, j * n + p);
            }
        }
    }

    // L * U, where L is unit lower triangular and U is upper triangular,
    // both stored in `lu`. Element (row i, col j) is at j * n + i.
    let mut prod = vec![0.0f64; n * n];
    for j in 0..n {
        for p in 0..=j {
            let u = lu[j * n + p] as f64; // U(p, j)
            prod[j * n + p] += u;         // L(p, p) = 1
            for i in (p + 1)..n {
                prod[j * n + i] += lu[p * n + i] as f64 * u; // L(i, p) * U(p, j)
            }
        }
    }

    let mut diff_sq = 0.0;
    let mut a_sq = 0.0;
    for idx in 0..n * n {
        let d = pa[idx] - prod[idx];
        diff_sq += d * d;
        a_sq += pa[idx] * pa[idx];
    }

    (diff_sq / a_sq).sqrt()
}

/// Compares two factorizations directly. Returns whether the pivots match
/// and the largest absolute elementwise difference between the factors.
pub fn compare_lu(lu_a: &[f32], piv_a: &[i32], lu_b: &[f32], piv_b: &[i32]) -> (bool, f32) {
    let pivots_match = piv_a == piv_b;
    let max_diff = lu_a
        .iter()
        .zip(lu_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    (pivots_match, max_diff)
}