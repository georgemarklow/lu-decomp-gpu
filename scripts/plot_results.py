
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# STEP 1: Configure paths and arguments
# ============================================

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

parser = argparse.ArgumentParser(
    description="Plot LU decomposition benchmark results."
)

parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help="Optional path to a specific benchmark CSV file."
)

args = parser.parse_args()


# ============================================
# STEP 2: Find the benchmark CSV
# ============================================

if args.csv is not None:

    csv_path = args.csv.resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Specified CSV file does not exist: {csv_path}"
        )

else:

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}\n"
            "Run your Rust benchmark first."
        )

    # Search all subdirectories for benchmark CSVs.
    csv_files = [
        path
        for path in results_dir.rglob("raw_lu*.csv")
        if path.is_file()
    ]

    print("\nBenchmark CSV files found:")

    for path in sorted(csv_files):
        print(f"  {path.relative_to(project_dir)}")

    if not csv_files:
        raise FileNotFoundError(
            f"No raw_lu*.csv files found inside {results_dir}.\n"
            "Check that your Rust benchmark calls save_result() "
            "with the correct CSV path."
        )

    # Choose the most recently modified CSV.
    csv_path = max(
        csv_files,
        key=lambda path: path.stat().st_mtime
    )


run_dir = csv_path.parent

figures_dir = run_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print(f"\nAnalysing CSV: {csv_path}")
print(f"Saving figures to: {figures_dir}")


# ============================================
# STEP 3: Read and validate CSV
# ============================================

df = pd.read_csv(csv_path)

required_columns = {
    "implementation",
    "matrix_size",
    "precision",
    "pivoting",
    "compute_ms",
    "status",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"CSV is missing columns: {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("The benchmark CSV contains no measurements.")

print("\nImplementations in CSV:")
print(df["implementation"].value_counts(dropna=False))

print("\nPrecision and pivoting combinations:")
print(
    df.groupby(
        ["implementation", "precision", "pivoting"],
        dropna=False
    ).size()
)


# ============================================
# STEP 4: Filter measurements
# ============================================

# Only successful benchmarks.
df = df[
    df["status"].astype(str).str.strip() == "success"
    ].copy()

# Select one experimental configuration.
# Adjust these to match your actual Rust benchmark.
selected_precision = "f32"
selected_pivoting = "partial"

df = df[
    (df["precision"].astype(str).str.strip()
     == selected_precision)
    &
    (df["pivoting"].astype(str).str.strip()
     == selected_pivoting)
    ].copy()

# Convert numerical columns safely.
df["compute_ms"] = pd.to_numeric(
    df["compute_ms"],
    errors="coerce"
)

df["matrix_size"] = pd.to_numeric(
    df["matrix_size"],
    errors="coerce"
)

df = df[
    df["compute_ms"].notna()
    & (df["compute_ms"] > 0)
    & df["matrix_size"].notna()
    & (df["matrix_size"] > 0)
    ].copy()

if df.empty:
    raise ValueError(
        "No valid benchmark measurements remain after filtering.\n"
        f"Selected precision: {selected_precision}\n"
        f"Selected pivoting: {selected_pivoting}\n"
        "Check the combinations printed above."
    )

df["matrix_size"] = df["matrix_size"].astype(int)

print("\nMeasurements after filtering:")
print(df.groupby("implementation").size())


# ============================================
# STEP 5: Calculate summary statistics
# ============================================

summary = (
    df.groupby(
        ["implementation", "matrix_size"],
        as_index=False
    )
    .agg(
        median_compute_ms=("compute_ms", "median"),
        mean_compute_ms=("compute_ms", "mean"),
        std_compute_ms=("compute_ms", "std"),
        min_compute_ms=("compute_ms", "min"),
        max_compute_ms=("compute_ms", "max"),
        trials=("compute_ms", "count"),
    )
)

summary["gflops"] = (
        (2.0 / 3.0)
        * summary["matrix_size"].astype(float) ** 3
        / (summary["median_compute_ms"] * 1e6)
)

print("\nBenchmark summary:")
print(summary.to_string(index=False))

summary.to_csv(
    run_dir / "summary.csv",
    index=False
)

print("\nSummary saved.")


# ============================================
# STEP 6: Reusable plotting function
# ============================================

def plot_metric(column, ylabel, title, filename):

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in summary.groupby("implementation"):

        data = data.sort_values("matrix_size")

        ax.plot(
            data["matrix_size"],
            data[column],
            marker="o",
            linewidth=2,
            label=name,
        )

    ax.set_xlabel("Matrix size (n × n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figures_dir / f"{filename}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        figures_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Generated: {filename}.pdf and .png")


# ============================================
# STEP 7: Plot execution time
# ============================================

plot_metric(
    column="median_compute_ms",
    ylabel="Median execution time (ms)",
    title="CPU vs GPU LU Execution Time",
    filename="execution_time",
)


# ============================================
# STEP 8: Plot effective GFLOP/s
# ============================================

plot_metric(
    column="gflops",
    ylabel="Effective GFLOP/s",
    title="CPU vs GPU LU Throughput",
    filename="gflops",
)


# ============================================
# STEP 9: Calculate GPU speedup
# ============================================

pivot = summary.pivot(
    index="matrix_size",
    columns="implementation",
    values="median_compute_ms",
)

required_implementations = {
    "lapack_cpu",
    "cubecl_gpu",
}

if required_implementations.issubset(pivot.columns):

    comparable = pivot[
        ["lapack_cpu", "cubecl_gpu"]
    ].dropna().copy()

    comparable = comparable[
        (comparable["lapack_cpu"] > 0)
        & (comparable["cubecl_gpu"] > 0)
        ].copy()

    if not comparable.empty:

        comparable["speedup"] = (
                comparable["lapack_cpu"]
                / comparable["cubecl_gpu"]
        )

        print("\nGPU speedup:")
        print(comparable.to_string())

        comparable.to_csv(
            run_dir / "speedup.csv"
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            comparable.index,
            comparable["speedup"],
            marker="o",
            linewidth=2,
            label="GPU speedup",
        )

        ax.axhline(
            y=1.0,
            linestyle="--",
            label="Equal performance",
        )

        ax.set_xlabel("Matrix size (n × n)")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.set_title("GPU Speedup Relative to LAPACK")

        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            figures_dir / "gpu_speedup.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            figures_dir / "gpu_speedup.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Generated: gpu_speedup.pdf and .png")

    else:
        print(
            "\nSkipping GPU speedup: "
            "no matching CPU/GPU matrix sizes."
        )

else:

    missing = required_implementations - set(pivot.columns)

    print(
        "\nSkipping GPU speedup."
        f"\nMissing implementation(s): {sorted(missing)}"
    )


# ============================================
# STEP 10: Finished
# ============================================

print("\nBenchmark analysis completed successfully!")
print(f"Results directory: {run_dir}")
print(f"Figures directory: {figures_dir}")
