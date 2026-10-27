
mod benchmark_csv;
use chrono::Utc;
use std::path::PathBuf;
use benchmark_csv::{save_result, BenchmarkResult};
use std::time::Instant;
use std::fs::create_dir_all;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // STEP 1: Generate a unique timestamp.
    let timestamp = Utc::now()
        .format("%Y-%m-%d_%H-%M-%S_%f")
        .to_string();

    // STEP 2: Create a unique folder name.
    let run_id = format!("run_{}", timestamp);

    // STEP 3: Define the folder path.
    let run_dir = PathBuf::from("results").join(&run_id);

    // STEP 4: Create the folder.
    create_dir_all(&run_dir)?;

    // STEP 5: Define the CSV file path.
    let csv_path = run_dir.join("raw_lu.csv");

    println!("Saving results to: {}", csv_path.display());
    // STEP 1: Choose matrix sizes to benchmark.
    let matrix_sizes = [128, 256, 512, 1024, 2048];

    // STEP 2: Number of timed repetitions.
    let repetitions = 10;

    // STEP 3: Loop through each matrix size.
    for &n in &matrix_sizes {

        println!("\nBenchmarking {} x {} matrix", n, n);

        // Generate one reproducible input matrix
        // for this matrix size.
        //let original_matrix = generate_matrix(n);

        // STEP 5: Run timed repetitions.
        for repetition in 1..=repetitions {

            // Restore the matrix before each run.
            //let mut matrix = original_matrix.clone();

            // Start timing.
            let start = Instant::now();

            // Execute your existing LAPACK LU routine.
            //run_lapack_lu(&mut matrix, n);

            // Stop timing.
            let compute_ms =
                start.elapsed().as_secs_f64() * 1000.0;

            println!(
                "n={}, repetition={}, time={:.4} ms",
                n, repetition, compute_ms
            );

            // STEP 6: Create CSV record.
            let record = BenchmarkResult {
                run_id: run_id.clone(),
                timestamp_utc: String::new(),
                trial_id: repetition as u32,
                implementation: "lapack_cpu".to_string(),
                backend: "openblas".to_string(),
                matrix_size: n,
                precision: "f32".to_string(),
                pivoting: "partial".to_string(),
                seed: 47290,
                repetition: repetition as u32,
                cpu_threads: Some(1),
                block_size: None,
                compute_ms,
                h2d_ms: None,
                d2h_ms: None,
                total_ms: compute_ms,
                residual: None,
                status: "success".to_string(),
            };

            println!("Saving benchmark results to: {}", csv_path.display());

            save_result(
                csv_path.to_str().expect("Invalid CSV path"),
                &record
            )?;
        }
    }

    println!("\nAll benchmarks completed!");

    Ok(())
}
