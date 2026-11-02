
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
# STEP 3b: Label each implementation
# ============================================

# The Rust benchmark has used several naming schemes:
#   implementation="lapack_cpu"  or "lapack"
#   implementation="cubecl"      with backend "cubecl", "cubecl_wgpu"
#                                or "cubecl_cpu"
#   implementation="cubecl_gpu_lu" / "cubecl_cpu_lu"
# Map them all onto three canonical labels.

def canonical_label(row):

    implementation = str(row["implementation"]).strip().lower()
    backend = str(row.get("backend", "")).strip().lower()

    if implementation.startswith("lapack"):
        return "lapack_cpu"

    if implementation.startswith("cubecl"):
        if "cpu" in implementation or "cpu" in backend:
            return "cubecl_cpu"
        return "cubecl_gpu"

    return implementation


df["label"] = df.apply(canonical_label, axis=1)

# If the GPU was run with several cube sizes, plot each one separately.
if "block_size" in df.columns:

    df["block_size"] = pd.to_numeric(
        df["block_size"],
        errors="coerce"
    )

    gpu_rows = df["label"] == "cubecl_gpu"
    gpu_block_sizes = df.loc[gpu_rows, "block_size"].dropna().unique()

    if len(gpu_block_sizes) > 1:
        df.loc[gpu_rows & df["block_size"].notna(), "label"] = (
            "cubecl_gpu_bs"
            + df.loc[gpu_rows & df["block_size"].notna(), "block_size"]
            .astype(int)
            .astype(str)
        )

display_names = {
    "lapack_cpu": "LAPACK (CPU, OpenBLAS)",
    "cubecl_gpu": "CubeCL (GPU, wgpu)",
    "cubecl_cpu": "CubeCL (CPU)",
}


def display_name(label):

    if label in display_names:
        return display_names[label]

    if label.startswith("cubecl_gpu_bs"):
        return f"CubeCL (GPU, {label.removeprefix('cubecl_gpu_bs')} threads/cube)"

    return label


print("\nLabels used for plotting:")
print(df.groupby(["implementation", "label"], dropna=False).size())


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

has_residual = "residual" in df.columns

if has_residual:
    df["residual"] = pd.to_numeric(
        df["residual"],
        errors="coerce"
    )

print("\nMeasurements after filtering:")
print(df.groupby("label").size())


# ============================================
# STEP 5: Calculate summary statistics
# ============================================

aggregations = dict(
    median_compute_ms=("compute_ms", "median"),
    mean_compute_ms=("compute_ms", "mean"),
    std_compute_ms=("compute_ms", "std"),
    min_compute_ms=("compute_ms", "min"),
    max_compute_ms=("compute_ms", "max"),
    trials=("compute_ms", "count"),
)

if has_residual:
    aggregations["median_residual"] = ("residual", "median")

summary = (
    df.groupby(
        ["label", "matrix_size"],
        as_index=False
    )
    .agg(**aggregations)
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

def plot_metric(column, ylabel, title, filename, log_y=False):

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in summary.groupby("label"):

        data = data.sort_values("matrix_size")

        if data[column].isna().all():
            continue

        ax.plot(
            data["matrix_size"],
            data[column],
            marker="o",
            linewidth=2,
            label=display_name(label),
        )

    ax.set_xlabel("Matrix size (n × n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Matrix sizes double each step, so a log2 x-axis spaces them evenly.
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["matrix_size"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    if log_y:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)
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

# Times span several orders of magnitude, so use a log y-axis.
plot_metric(
    column="median_compute_ms",
    ylabel="Median execution time (ms)",
    title="LU Execution Time: LAPACK vs CubeCL (CPU and GPU)",
    filename="execution_time",
    log_y=True,
)


# ============================================
# STEP 8: Plot effective GFLOP/s
# ============================================

plot_metric(
    column="gflops",
    ylabel="Effective GFLOP/s",
    title="LU Throughput: LAPACK vs CubeCL (CPU and GPU)",
    filename="gflops",
    log_y=True,
)


# ============================================
# STEP 8b: Plot residuals
# ============================================

if has_residual and summary["median_residual"].notna().any():

    plot_metric(
        column="median_residual",
        ylabel="Relative residual ||PA − LU||_F / ||A||_F",
        title="LU Accuracy: LAPACK vs CubeCL (CPU and GPU)",
        filename="residual",
        log_y=True,
    )

else:
    print("\nSkipping residual plot: no residual values in CSV.")


# ============================================
# STEP 9: Calculate speedup relative to LAPACK
# ============================================

pivot = summary.pivot(
    index="matrix_size",
    columns="label",
    values="median_compute_ms",
)

baseline = "lapack_cpu"

# Every CubeCL variant present (GPU, CPU, and any GPU cube sizes).
compared = sorted(
    label
    for label in pivot.columns
    if label.startswith("cubecl")
)

if baseline in pivot.columns and compared:

    speedup = pd.DataFrame(index=pivot.index)

    for label in compared:
        valid = (pivot[baseline] > 0) & (pivot[label] > 0)
        speedup[label] = (pivot[baseline] / pivot[label]).where(valid)

    speedup = speedup.dropna(how="all")

    if not speedup.empty:

        print("\nSpeedup relative to LAPACK (LAPACK time / CubeCL time):")
        print(speedup.to_string())

        speedup.to_csv(
            run_dir / "speedup.csv"
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        for label in compared:

            series = speedup[label].dropna()

            if series.empty:
                continue

            ax.plot(
                series.index,
                series,
                marker="o",
                linewidth=2,
                label=display_name(label),
            )

        ax.axhline(
            y=1.0,
            color="grey",
            linestyle="--",
            label="Equal to LAPACK",
        )

        ax.set_xlabel("Matrix size (n × n)")
        ax.set_ylabel("Speedup (LAPACK time / CubeCL time)")
        ax.set_title("CubeCL Speedup Relative to LAPACK")

        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(speedup.index))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Values below 1 (slower than LAPACK) are easier to read on a log axis.
        ax.set_yscale("log")

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            figures_dir / "speedup.pdf",
            bbox_inches="tight"
        )

        fig.savefig(
            figures_dir / "speedup.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Generated: speedup.pdf and .png")

    else:
        print(
            "\nSkipping speedup: "
            "no matrix sizes shared by LAPACK and CubeCL."
        )

else:

    missing = []

    if baseline not in pivot.columns:
        missing.append(baseline)

    if not compared:
        missing.append("cubecl_gpu / cubecl_cpu")

    print(
        "\nSkipping speedup."
        f"\nMissing implementation(s): {missing}"
    )


# ============================================
# STEP 10: Finished
# ============================================

print("\nBenchmark analysis completed successfully!")
print(f"Results directory: {run_dir}")
print(f"Figures directory: {figures_dir}")
