import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS to your actual metadata CSV path
CSV_PATH = Path(r"G:/My Drive/DeepfakeResearch/features/metadata/metadata_features.csv")

# Analysis output directory
OUTPUT_DIR = Path(r"G:/My Drive/DeepfakeResearch/outputs/metadata_analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"
CAT_DIR = OUTPUT_DIR / "categorical_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "file_size",
    "width",
    "height",
    "aspect_ratio",
    "duration",
    "fps",
]

CATEGORICAL_COLS = [
    "file_extension",
    "media_type",
    "codec",
    "encoder",
    "camera_make",
    "camera_model",
    "software",
    "container_format",
]

FLAG_COLS = [
    "has_exif",
    "has_gps",
    "color_channels",
    "bit_depth",
]


# ============================================================
# LOAD DATA
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Metadata CSV not found: {CSV_PATH}\n"
        "Please edit CSV_PATH at the top of this file."
    )

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("DEEPFAKE METADATA EXPLORATORY ANALYSIS")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

required = ["label", "dataset"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False))

print("\nDataset distribution:")
print(df["dataset"].value_counts(dropna=False))

print("\nDataset x Label:")
print(pd.crosstab(df["dataset"], df["label"]))


# ============================================================
# MISSING / UNKNOWN ANALYSIS
# ============================================================

# Treat textual Unknown values as missing for data-quality analysis.
analysis_df = df.replace(
    ["Unknown", "unknown", "UNKNOWN", "", "NA", "N/A", "null", "None"],
    np.nan
)

missing_rows = []

for col in analysis_df.columns:
    null_count = analysis_df[col].isna().sum()

    numeric_version = pd.to_numeric(
        analysis_df[col], errors="coerce"
    )
    minus_one_count = (numeric_version == -1).sum()

    missing_rows.append({
        "feature": col,
        "null_or_unknown_count": int(null_count),
        "null_or_unknown_percentage":
            round(100 * null_count / len(df), 2),
        "minus_one_count": int(minus_one_count),
        "minus_one_percentage":
            round(100 * minus_one_count / len(df), 2),
        "unique_values":
            int(analysis_df[col].nunique(dropna=False)),
    })

missing_df = pd.DataFrame(missing_rows)
missing_df = missing_df.sort_values(
    "null_or_unknown_percentage",
    ascending=False
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)

print("\nMissing / Unknown analysis:")
print(missing_df.to_string(index=False))


# ============================================================
# UNIQUE VALUES
# ============================================================

with open(
    OUTPUT_DIR / "summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("DEEPFAKE METADATA ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Rows: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n\n")

    f.write("LABEL DISTRIBUTION\n")
    f.write(str(df["label"].value_counts(dropna=False)))
    f.write("\n\n")

    f.write("DATASET DISTRIBUTION\n")
    f.write(str(df["dataset"].value_counts(dropna=False)))
    f.write("\n\n")

    f.write("DATASET x LABEL\n")
    f.write(str(pd.crosstab(df["dataset"], df["label"])))
    f.write("\n\n")

    f.write("UNIQUE VALUES\n")

    for col in df.columns:
        f.write(f"\n--- {col} ---\n")
        f.write(
            str(df[col].value_counts(dropna=False).head(30))
        )
        f.write("\n")


# ============================================================
# NUMERICAL FEATURES
# ============================================================

available_numeric = [
    c for c in NUMERIC_COLS
    if c in df.columns
]

for col in available_numeric:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

overall_stats = df[available_numeric].describe().T

overall_stats["missing_count"] = (
    df[available_numeric].isna().sum()
)

overall_stats["missing_percentage"] = (
    df[available_numeric].isna().mean() * 100
)

overall_stats.to_csv(
    OUTPUT_DIR / "numeric_overall_statistics.csv"
)

print("\nOverall numerical statistics:")
print(overall_stats)


# ============================================================
# REAL VS FAKE NUMERICAL STATISTICS
# ============================================================

real_fake_stats = (
    df.groupby("label")[available_numeric]
      .agg(["count", "mean", "median", "std", "min", "max"])
      .T
)

real_fake_stats.to_csv(
    OUTPUT_DIR / "numeric_real_vs_fake.csv"
)

print("\nReal vs Fake numerical statistics:")
print(real_fake_stats)


# ============================================================
# REAL VS FAKE BOXPLOTS
# ============================================================

for col in available_numeric:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="label",
        y=col
    )

    plt.title(f"{col}: Real vs Fake")
    plt.xlabel("Label")
    plt.ylabel(col)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{col}_real_vs_fake_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# DATASET-WISE NUMERICAL ANALYSIS
# ============================================================

dataset_stats = (
    df.groupby(["dataset", "label"])[available_numeric]
      .agg(["mean", "median", "std"])
)

dataset_stats.to_csv(
    OUTPUT_DIR / "numeric_dataset_label_statistics.csv"
)

print("\nDataset-wise numerical statistics:")
print(dataset_stats)


for col in available_numeric:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="dataset",
        y=col,
        hue="label"
    )

    plt.title(f"{col}: Dataset and Label Comparison")
    plt.xlabel("Dataset")
    plt.ylabel(col)
    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"{col}_dataset_label_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_rows = []

for col in CATEGORICAL_COLS:

    if col not in df.columns:
        continue

    counts = df[col].value_counts(dropna=False)

    for value, count in counts.items():

        categorical_rows.append({
            "feature": col,
            "value": value,
            "count": int(count),
            "percentage":
                round(100 * count / len(df), 2)
        })

categorical_counts = pd.DataFrame(
    categorical_rows
)

categorical_counts.to_csv(
    OUTPUT_DIR / "categorical_value_counts.csv",
    index=False
)


# ============================================================
# CATEGORICAL REAL/FAKE ANALYSIS
# CHI-SQUARE TEST
# ============================================================

chi_results = []

for col in CATEGORICAL_COLS:

    if col not in df.columns:
        continue

    table = pd.crosstab(
        df[col].fillna("MISSING"),
        df["label"]
    )

    table.to_csv(
        CAT_DIR / f"{col}_real_vs_fake.csv"
    )

    if table.shape[0] >= 2 and table.shape[1] >= 2:

        chi2, p_value, dof, expected = chi2_contingency(table)

        chi_results.append({
            "feature": col,
            "chi2": chi2,
            "degrees_of_freedom": dof,
            "p_value": p_value,
            "number_of_categories": table.shape[0],
        })

    if table.shape[0] <= 15:

        percentage_table = (
            pd.crosstab(
                df[col].fillna("MISSING"),
                df["label"],
                normalize="columns"
            ) * 100
        )

        ax = percentage_table.plot(
            kind="bar",
            figsize=(9, 6)
        )

        ax.set_title(
            f"{col}: Real vs Fake Distribution"
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Percentage")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{col}_real_vs_fake.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


chi_df = pd.DataFrame(chi_results)

if not chi_df.empty:
    chi_df = chi_df.sort_values("p_value")

chi_df.to_csv(
    OUTPUT_DIR / "categorical_chi_square_tests.csv",
    index=False
)

print("\nChi-square results:")
print(chi_df)


# ============================================================
# CORRELATION WITH REAL/FAKE
# ============================================================

df["label_binary"] = (
    df["label"]
      .astype(str)
      .str.lower()
      .map({
          "real": 0,
          "fake": 1
      })
)

correlation = (
    df[available_numeric + ["label_binary"]]
      .corr(numeric_only=True)["label_binary"]
      .drop("label_binary")
      .sort_values(
          key=lambda x: x.abs(),
          ascending=False
      )
)

correlation.to_csv(
    OUTPUT_DIR / "correlation_with_label.csv"
)

print("\nCorrelation with REAL/FAKE label:")
print(correlation)


plt.figure(figsize=(9, 7))

sns.heatmap(
    df[available_numeric + ["label_binary"]].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Numerical Metadata Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "numeric_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

test_results = []

for col in available_numeric:

    real_values = df.loc[
        df["label"] == "real", col
    ].dropna()

    fake_values = df.loc[
        df["label"] == "fake", col
    ].dropna()

    if len(real_values) == 0 or len(fake_values) == 0:
        continue

    statistic, p_value = mannwhitneyu(
        real_values,
        fake_values,
        alternative="two-sided"
    )

    test_results.append({
        "feature": col,
        "real_n": len(real_values),
        "fake_n": len(fake_values),
        "real_median": real_values.median(),
        "fake_median": fake_values.median(),
        "U_statistic": statistic,
        "p_value": p_value,
        "significant_at_0.05":
            p_value < 0.05
    })

tests_df = pd.DataFrame(test_results)

if not tests_df.empty:
    tests_df = tests_df.sort_values("p_value")

tests_df.to_csv(
    OUTPUT_DIR / "statistical_tests.csv",
    index=False
)

print("\nMann-Whitney U test:")
print(tests_df)


# ============================================================
# DATASET/LABEL COUNTS
# ============================================================

pd.crosstab(
    df["dataset"],
    df["label"]
).to_csv(
    OUTPUT_DIR / "dataset_label_counts.csv"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nResults saved to:")
print(OUTPUT_DIR)

print("\nMain outputs:")
print("  missing_values.csv")
print("  numeric_overall_statistics.csv")
print("  numeric_real_vs_fake.csv")
print("  numeric_dataset_label_statistics.csv")
print("  categorical_value_counts.csv")
print("  categorical_chi_square_tests.csv")
print("  correlation_with_label.csv")
print("  statistical_tests.csv")
print("  dataset_label_counts.csv")
print("  summary.txt")

print("\nPlots:")
print(PLOTS_DIR)

print("\nIMPORTANT:")
print("Do not remove features only because correlation or p-value is low.")
print("The next stage should be a metadata-only classifier baseline.")
print("Cross-dataset testing is also important because dataset-specific")
print("encoding patterns may otherwise create misleading results.")