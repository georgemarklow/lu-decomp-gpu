/// Computes the LU factorization of an n x n column-major matrix in place,
/// using LAPACK sgetrf with partial pivoting.
///
/// After the call, `matrix` holds U in its upper triangle and L (whose
/// diagonal of 1s is not stored) in its strict lower triangle.
/// The returned pivot indices are 1-based, as in Fortran.
pub fn run_lapack_lu(matrix: &mut [f32], n: usize) -> Vec<i32> {
    assert_eq!(matrix.len(), n * n, "matrix must be n x n");

    let n_i32 = n as i32;
    let mut ipiv = vec![0i32; n];
    let mut info = 0;

    unsafe {
        lapack::sgetrf(n_i32, n_i32, matrix, n_i32, &mut ipiv, &mut info);
    }

    if info < 0 {
        panic!("sgetrf: argument {} had an illegal value", -info);
    } else if info > 0 {
        eprintln!("sgetrf: U({0},{0}) is exactly zero; matrix is singular", info);
    }

    ipiv
}