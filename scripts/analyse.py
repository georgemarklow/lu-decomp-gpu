
from pathlib import Path
import argparse

import pandas as pd


# ============================================
# STEP 1: Configure paths
# ============================================

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

parser = argparse.ArgumentParser(
    description="Analyse LU decomposition benchmark results."
)

parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help="Path to a specific benchmark CSV file."
)

args = parser.parse_args()


# ============================================
# STEP 2: Find benchmark CSV file
# ============================================

if args.csv is not None:

    # Use the CSV explicitly selected by the user.
    csv_path = args.csv.resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

else:

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}"
        )

    # Find benchmark CSV files in all run folders.
    csv_files = [
        path
        for path in results_dir.rglob("raw_lu*.csv")
        if path.is_file()
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No benchmark CSV files found in {results_dir}"
        )

    # Select the most recently modified CSV.
    csv_path = max(
        csv_files,
        key=lambda path: path.stat().st_mtime
    )


# Use the selected CSV's parent directory.
run_dir = csv_path.parent

print(f"\nAnalysing: {csv_path}")


# ============================================
# STEP 3: Read benchmark results
# ============================================

df = pd.read_csv(csv_path)

required_columns = {
    "implementation",
    "matrix_size",
    "precision",
    "pivoting",
    "compute_ms",
    "total_ms",
    "trial_id",
    "status",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing CSV columns: {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("The benchmark CSV is empty.")


# ============================================
# STEP 4: Filter valid measurements
# ============================================

# Only include successful benchmark trials.
df = df[
    df["status"].astype(str).str.strip() == "success"
    ].copy()

# Convert timing values to numeric.
df["compute_ms"] = pd.to_numeric(
    df["compute_ms"],
    errors="coerce"
)

df["total_ms"] = pd.to_numeric(
    df["total_ms"],
    errors="coerce"
)

df["matrix_size"] = pd.to_numeric(
    df["matrix_size"],
    errors="coerce"
)

# Exclude invalid compute measurements.
df = df[
    df["compute_ms"].notna()
    & (df["compute_ms"] > 0)
    & df["matrix_size"].notna()
    & (df["matrix_size"] > 0)
    ].copy()

if df.empty:
    raise ValueError(
        "No valid successful benchmark measurements found."
    )

df["matrix_size"] = df["matrix_size"].astype(int)


# ============================================
# STEP 5: Calculate summary statistics
# ============================================

summary = (
    df.groupby(
        [
            "implementation",
            "matrix_size",
            "precision",
            "pivoting",
        ],
        dropna=False,
    )
    .agg(
        median_compute_ms=("compute_ms", "median"),
        median_total_ms=("total_ms", "median"),
        mean_compute_ms=("compute_ms", "mean"),
        std_compute_ms=("compute_ms", "std"),
        min_compute_ms=("compute_ms", "min"),
        max_compute_ms=("compute_ms", "max"),
        trials=("trial_id", "count"),
    )
    .reset_index()
    .sort_values(["implementation", "matrix_size"])
)


# ============================================
# STEP 6: Save summary CSV
# ============================================

summary_path = run_dir / "summary.csv"

summary.to_csv(
    summary_path,
    index=False
)


# ============================================
# STEP 7: Display results
# ============================================

print("\nBenchmark summary:")
print(summary.to_string(index=False))

print(f"\nSummary saved to: {summary_path}")
print("\nAnalysis completed successfully!")
