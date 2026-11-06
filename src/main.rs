extern crate rand;
extern crate lapack_src;

mod benchmark_csv;
mod cubecl_cpu_lu;
mod cubecl_gpu_lu;
pub mod lapack_lu;
mod utilities;

use chrono::Utc;
use std::path::PathBuf;
use benchmark_csv::{save_result, BenchmarkResult};
use std::time::Instant;
use std::fs::create_dir_all;
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::cubecl_gpu_lu::run_cubecl_gpu_lu;
use crate::cubecl_cpu_lu::run_cubecl_cpu_lu;
use crate::lapack_lu::run_lapack_lu;
use crate::utilities::{compare_lu, lu_residual};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    const SEED: u64 = 42;

    // CPU thread counts to benchmark for the CubeCL CPU runtime's trailing update.
    const CPU_THREAD_COUNTS: [u32; 4] = [1, 2, 4, 8];

    // Threads per cube to benchmark for the CubeCL GPU kernels
    // (each must be a power of two, at most the device limit).
    const GPU_THREAD_COUNTS: [u32; 5] = [16, 32, 64, 128, 256];

    let timestamp = Utc::now()
        .format("%Y-%m-%d_%H-%M-%S_%f")
        .to_string();

    let run_id = format!("run_{}", timestamp);

    let run_dir = PathBuf::from("results").join(&run_id);

    create_dir_all(&run_dir)?;

    let csv_path = run_dir.join("raw_lu.csv");

    println!("Saving results to: {}", csv_path.display());

    // Choose matrix sizes to benchmark.
    let matrix_sizes = [128, 256, 512, 1024, 2048];

    // How many times to run the same computation.
    let repetitions = 10;

    for &n in &matrix_sizes {

        println!("\nBenchmarking {} x {} matrix", n, n);

        // Generate one reproducible input matrix
        // for this matrix size.
        let original_matrix = generate_matrix(n, SEED);

        // Warm up each CubeCL configuration once (untimed), so kernel
        // compilation for each thread count is not counted in repetition 1.
        for &threads in &CPU_THREAD_COUNTS {
            run_cubecl_cpu_lu(&mut original_matrix.clone(), n, threads);
        }
        for &threads in &GPU_THREAD_COUNTS {
            run_cubecl_gpu_lu(&mut original_matrix.clone(), n, threads);
        }

        // Run timed repetitions.
        for repetition in 1..=repetitions {

            ///////////////////////////
            // LAPACK Benchmarking
            ///////////////////////////

            // Restore the matrix before each run.
            let mut lapack_matrix = original_matrix.clone();

            // Start timing.
            let start = Instant::now();

            // Execute your existing LAPACK LU routine.
            let lapack_piv = run_lapack_lu(&mut lapack_matrix, n);

            // Stop timing.
            let compute_ms =
                start.elapsed().as_secs_f64() * 1000.0;

            // Compute LAPACK residual
            let lapack_residual = lu_residual(&original_matrix, &lapack_matrix, &lapack_piv, n);

            println!(
                "n={}, repetition={}, time={:.4} ms",
                n, repetition, compute_ms
            );

            // Create CSV record.
            let record = BenchmarkResult {
                run_id: run_id.clone(),
                timestamp_utc: String::new(),
                trial_id: repetition as u32,
                implementation: "LAPACK CPU".to_string(),
                backend: "LAPACK CPU".to_string(),
                matrix_size: n,
                precision: "f32".to_string(),
                pivoting: "partial".to_string(),
                seed: SEED,
                repetition: repetition as u32,
                cpu_threads: Some(1),
                block_size: None,
                compute_ms,
                h2d_ms: None,
                d2h_ms: None,
                total_ms: compute_ms,
                residual: Some(lapack_residual),
                status: "success".to_string(),
            };

            println!("Saving benchmark results to: {}", csv_path.display());

            save_result(
                csv_path.to_str().expect("Invalid CSV path"),
                &record
            )?;

            println!("  LAPACK residual: {:.3e}", lapack_residual);

            ///////////////////////////
            // CubeCL CPU Benchmarking
            ///////////////////////////

            for &threads in &CPU_THREAD_COUNTS {

                let label = format!("CUBECL CPU ({} threads)", threads);

                // Restore the matrix before each run.
                let mut cpu_matrix = original_matrix.clone();

                // Start timing.
                let start = Instant::now();

                // Execute CubeCL on the CPU runtime.
                let cpu_piv = run_cubecl_cpu_lu(&mut cpu_matrix, n, threads);

                // Stop timing.
                let compute_ms =
                    start.elapsed().as_secs_f64() * 1000.0;

                let cpu_residual = lu_residual(&original_matrix, &cpu_matrix, &cpu_piv, n);

                // Verify against LAPACK.
                let (pivots_match, max_diff) =
                    compare_lu(&lapack_matrix, &lapack_piv, &cpu_matrix, &cpu_piv);

                println!(
                    "n={}, repetition={}, {}: time={:.4} ms, residual={:.3e}, pivots match={}, max |LAPACK - CubeCL|={:.3e}",
                    n, repetition, label, compute_ms, cpu_residual, pivots_match, max_diff
                );

                if !pivots_match {
                    println!("\nPivots do not match");
                    return Err("Pivots do not match".into());
                }

                // Create CSV record.
                let cubecl_cpu_record = BenchmarkResult {
                    run_id: run_id.clone(),
                    timestamp_utc: String::new(),
                    trial_id: repetition as u32,
                    implementation: label.clone(),
                    backend: "CUBECL CPU".to_string(),
                    matrix_size: n,
                    precision: "f32".to_string(),
                    pivoting: "partial".to_string(),
                    seed: SEED,
                    repetition: repetition as u32,
                    cpu_threads: Some(threads),
                    block_size: None,
                    compute_ms,
                    h2d_ms: None,
                    d2h_ms: None,
                    total_ms: compute_ms,
                    residual: Some(cpu_residual),
                    status: "success".to_string(),
                };

                save_result(
                    csv_path.to_str().expect("Invalid CSV path"),
                    &cubecl_cpu_record
                )?;
            }

            ///////////////////////////
            // CubeCL GPU Benchmarking
            ///////////////////////////

            for &threads in &GPU_THREAD_COUNTS {

                let label = format!("CUBECL GPU ({} threads)", threads);

                // Restore the matrix before each run.
                let mut gpu_matrix = original_matrix.clone();

                // Start timing.
                let start = Instant::now();

                // Execute CubeCL on the GPU (wgpu / Metal).
                let gpu_piv = run_cubecl_gpu_lu(&mut gpu_matrix, n, threads);

                // Stop timing.
                let compute_ms =
                    start.elapsed().as_secs_f64() * 1000.0;

                let gpu_residual = lu_residual(&original_matrix, &gpu_matrix, &gpu_piv, n);

                // Verify against LAPACK.
                let (pivots_match, max_diff) =
                    compare_lu(&lapack_matrix, &lapack_piv, &gpu_matrix, &gpu_piv);

                println!(
                    "n={}, repetition={}, {}: time={:.4} ms, residual={:.3e}, pivots match={}, max |LAPACK - CubeCL|={:.3e}",
                    n, repetition, label, compute_ms, gpu_residual, pivots_match, max_diff
                );

                if !pivots_match {
                    println!("\nPivots do not match");
                    return Err("Pivots do not match".into());
                }

                // Create CSV record.
                let cubecl_gpu_record = BenchmarkResult {
                    run_id: run_id.clone(),
                    timestamp_utc: String::new(),
                    trial_id: repetition as u32,
                    implementation: label.clone(),
                    backend: "CUBECL GPU".to_string(),
                    matrix_size: n,
                    precision: "f32".to_string(),
                    pivoting: "partial".to_string(),
                    seed: SEED,
                    repetition: repetition as u32,
                    cpu_threads: None,
                    block_size: Some(threads as usize),
                    compute_ms,
                    h2d_ms: None,
                    d2h_ms: None,
                    total_ms: compute_ms,
                    residual: Some(gpu_residual),
                    status: "success".to_string(),
                };

                save_result(
                    csv_path.to_str().expect("Invalid CSV path"),
                    &cubecl_gpu_record
                )?;
            }
        }
    }

    println!("\nAll benchmarks completed!");

    Ok(())
}

/// Generates an n x n matrix in column-major order with values in [0, 1).
/// Element (row, col) is stored at index col * n + row.
fn generate_matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut matrix = vec![0.0f32; n * n];

    for col in 0..n {
        for row in 0..n {
            matrix[col * n + row] = rng.random_range(0.0..1.0);
        }
    }

    matrix
}