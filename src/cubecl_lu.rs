/// Factors an n x n column-major matrix in place with partial pivoting.
/// Returns 1-based pivot indices, like LAPACK sgetrf.
/// Element A[row, col] is stored at index col * n + row.
pub fn lu_cpu(matrix: &mut [f32], n: usize) -> Vec<i32> {
    let mut piv = vec![0u32; n];

    for k in 0..n {
        // 1. Find the row with the largest |A[i, k]| for i >= k.
        let mut best = k;
        let mut best_val = matrix[k * n + k].abs();
        for i in k + 1..n {
            let v = matrix[k * n + i].abs();
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        piv[k] = best as u32;

        // 2. Swap rows k and best across all columns.
        if best != k {
            for j in 0..n {
                matrix.swap(j * n + k, j * n + best);
            }
        }

        let pivot = matrix[k * n + k];
        if pivot != 0.0 {
            // 3. Compute column k of L.
            for i in k + 1..n {
                matrix[k * n + i] /= pivot;
            }

            // 4. Update the trailing submatrix.
            for j in k + 1..n {
                let u = matrix[j * n + k];
                for i in k + 1..n {
                    matrix[j * n + i] -= matrix[k * n + i] * u;
                }
            }
        }
    }

    piv.iter().map(|&p| p as i32 + 1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lapack_lu::run_lapack_lu;

    #[test]
    fn test_lu_cpu() {
        // A = | 2   1  1 |
        //     | 4  -6  0 |
        //     |-2   7  2 |
        // stored column-major: element (row, col) is at col * n + row.
        let n = 3;
        let matrix_orig: Vec<f32> = vec![
            2.0, 4.0, -2.0, // column 0
            1.0, -6.0, 7.0, // column 1
            1.0, 0.0, 2.0,  // column 2
        ];

        // Check what LAPACK returns
        let mut lapack_matrix = matrix_orig.clone();
        let lapack_pivots = run_lapack_lu(&mut lapack_matrix, n);

        let mut matrix = matrix_orig.clone();
        let pivots = lu_cpu(&mut matrix, n);

        // Pivots and LU factors should match LAPACK
        assert_eq!(pivots, lapack_pivots);
        assert!((matrix.iter().zip(lapack_matrix.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)) < 1e-5);
    }
}