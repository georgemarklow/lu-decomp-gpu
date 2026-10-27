
use serde::Serialize;
use std::error::Error;
use std::fs::{create_dir_all, OpenOptions};
use std::path::Path;

#[derive(Serialize)]
pub struct BenchmarkResult {
    pub run_id: String,
    pub timestamp_utc: String,
    pub trial_id: u32,
    pub implementation: String,
    pub backend: String,
    pub matrix_size: usize,
    pub precision: String,
    pub pivoting: String,
    pub seed: u64,
    pub repetition: u32,
    pub cpu_threads: Option<u32>,
    pub block_size: Option<usize>,
    pub compute_ms: f64,
    pub h2d_ms: Option<f64>,
    pub d2h_ms: Option<f64>,
    pub total_ms: f64,
    pub residual: Option<f64>,
    pub status: String,
}

pub fn save_result(
    path: &str,
    result: &BenchmarkResult,
) -> Result<(), Box<dyn Error>> {
    let path = Path::new(path);

    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    let write_header =
        !path.exists() || path.metadata()?.len() == 0;

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    let mut writer = csv::WriterBuilder::new()
        .has_headers(write_header)
        .from_writer(file);

    writer.serialize(result)?;
    writer.flush()?;

    Ok(())
}
