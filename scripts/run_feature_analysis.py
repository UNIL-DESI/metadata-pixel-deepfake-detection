import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"G:\My Drive\DeepfakeResearch"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "features",
    "aligned"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "features",
    "analysis"
)

HANDCRAFTED_FILE = os.path.join(
    INPUT_DIR,
    "aligned_handcrafted.csv"
)

CNN_FILE = os.path.join(
    INPUT_DIR,
    "aligned_cnn.csv"
)

METADATA_FILE = os.path.join(
    INPUT_DIR,
    "aligned_metadata.csv"
)

ALL_FEATURES_FILE = os.path.join(
    INPUT_DIR,
    "aligned_all_features.csv"
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

RANDOM_STATE = 42

# Variance threshold for exploratory near-zero variance check
VARIANCE_THRESHOLD = 1e-8

# Correlation threshold
CORRELATION_THRESHOLD = 0.95

# Number of RF features to report
TOP_RF_FEATURES = 50

# Number of mutual-information features to report
TOP_MI_FEATURES = 50


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_feature_file(path, name):
    print(f"\nLoading {name}...")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"{name} shape: {df.shape}")

    return df


def identify_columns(df):
    """
    Identify label and identifier columns.
    Everything else is treated as a feature.
    """

    identifier_candidates = [
        "media_id",
        "video_id",
        "video_name"
    ]

    id_columns = [
        col for col in identifier_candidates
        if col in df.columns
    ]

    label_column = "label" if "label" in df.columns else None

    feature_columns = [
        col for col in df.columns
        if col not in id_columns
        and col != label_column
    ]

    return id_columns, label_column, feature_columns


def convert_labels(y):
    """
    Convert real/fake labels into binary values for analysis.
    real = 0
    fake = 1
    """

    y = y.astype(str).str.lower().str.strip()

    mapping = {
        "real": 0,
        "fake": 1,
        "0": 0,
        "1": 1
    }

    converted = y.map(mapping)

    if converted.isna().any():
        invalid = sorted(y[converted.isna()].unique())

        raise ValueError(
            f"Unexpected labels found: {invalid}"
        )

    return converted.astype(int)


def prepare_features(df):
    """
    Return X and y.
    """

    id_columns, label_column, feature_columns = identify_columns(df)

    if label_column is None:
        raise ValueError(
            "Dataset does not contain a 'label' column."
        )

    X = df[feature_columns].copy()
    y = convert_labels(df[label_column])

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, id_columns, label_column, feature_columns


# ============================================================
# BASIC DATA QUALITY ANALYSIS
# ============================================================

def analyze_data_quality(X, name):
    print_section(f"{name} - DATA QUALITY")

    results = {}

    results["samples"] = len(X)
    results["features"] = X.shape[1]

    missing_total = int(X.isna().sum().sum())

    results["missing_values"] = missing_total

    print(f"Samples:         {results['samples']}")
    print(f"Features:        {results['features']}")
    print(f"Missing values:  {missing_total}")

    # Infinite values
    infinite_count = int(
        np.isinf(X.select_dtypes(include=np.number)).sum().sum()
    )

    results["infinite_values"] = infinite_count

    print(f"Infinite values: {infinite_count}")

    return results


# ============================================================
# CONSTANT / LOW VARIANCE FEATURES
# ============================================================

def analyze_variance(X, name):
    print_section(f"{name} - VARIANCE ANALYSIS")

    variance = X.var(axis=0, skipna=True)

    constant_features = variance[
        variance == 0
    ].index.tolist()

    near_zero_features = variance[
        variance <= VARIANCE_THRESHOLD
    ].index.tolist()

    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )

    print(
        f"Near-zero variance features: "
        f"{len(near_zero_features)}"
    )

    return variance, constant_features, near_zero_features


# ============================================================
# DUPLICATE FEATURES
# ============================================================

def find_duplicate_features(X, name):
    print_section(f"{name} - DUPLICATE FEATURE ANALYSIS")

    duplicates = []

    columns = X.columns

    # Transpose allows duplicated() to compare columns.
    duplicate_mask = X.T.duplicated()

    duplicates = columns[duplicate_mask].tolist()

    print(
        f"Duplicate feature columns: "
        f"{len(duplicates)}"
    )

    return duplicates


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def analyze_correlation(X, name):
    print_section(f"{name} - CORRELATION ANALYSIS")

    if X.shape[1] == 0:
        return [], pd.DataFrame()

    print(
        f"Calculating correlation matrix "
        f"for {X.shape[1]} features..."
    )

    correlation = X.corr().abs()

    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1
        ).astype(bool)
    )

    highly_correlated = []

    for column in upper.columns:

        correlated_columns = upper.index[
            upper[column] > CORRELATION_THRESHOLD
        ].tolist()

        for row in correlated_columns:

            highly_correlated.append(
                {
                    "feature_1": row,
                    "feature_2": column,
                    "correlation": upper.loc[row, column]
                }
            )

    correlation_pairs = pd.DataFrame(
        highly_correlated
    )

    print(
        f"Highly correlated pairs "
        f"(>{CORRELATION_THRESHOLD}): "
        f"{len(correlation_pairs)}"
    )

    return highly_correlated, correlation_pairs


# ============================================================
# CLASS-WISE FEATURE ANALYSIS
# ============================================================

def analyze_class_difference(X, y, name):
    print_section(f"{name} - CLASS-WISE ANALYSIS")

    real_mask = y == 0
    fake_mask = y == 1

    real_mean = X.loc[real_mask].mean()
    fake_mean = X.loc[fake_mask].mean()

    difference = (
        fake_mean - real_mean
    ).abs()

    result = pd.DataFrame({
        "real_mean": real_mean,
        "fake_mean": fake_mean,
        "absolute_difference": difference
    })

    result = result.sort_values(
        "absolute_difference",
        ascending=False
    )

    print(
        f"Top {min(20, len(result))} "
        f"class-different features:"
    )

    print(
        result.head(20).to_string()
    )

    return result


# ============================================================
# RANDOM FOREST IMPORTANCE
# ============================================================

def analyze_random_forest(X, y, name):
    print_section(
        f"{name} - PRELIMINARY RANDOM FOREST IMPORTANCE"
    )

    print(
        "Training exploratory Random Forest..."
    )

    # Replace missing values
    X_rf = X.copy()

    X_rf = X_rf.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_rf = X_rf.fillna(
        X_rf.median()
    )

    # Remove constant features
    variance = X_rf.var()

    valid_columns = variance[
        variance > VARIANCE_THRESHOLD
    ].index

    X_rf = X_rf[
        valid_columns
    ]

    print(
        f"Features used by RF: "
        f"{X_rf.shape[1]}"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X_rf,
        y
    )

    importance = pd.DataFrame({
        "feature": X_rf.columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print(
        f"\nTop {TOP_RF_FEATURES} "
        f"Random Forest features:"
    )

    print(
        importance.head(
            TOP_RF_FEATURES
        ).to_string(index=False)
    )

    return importance


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def analyze_mutual_information(X, y, name):
    print_section(
        f"{name} - MUTUAL INFORMATION"
    )

    X_mi = X.copy()

    X_mi = X_mi.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_mi = X_mi.fillna(
        X_mi.median()
    )

    variance = X_mi.var()

    valid_columns = variance[
        variance > VARIANCE_THRESHOLD
    ].index

    X_mi = X_mi[
        valid_columns
    ]

    print(
        f"Calculating mutual information "
        f"for {X_mi.shape[1]} features..."
    )

    mi_scores = mutual_info_classif(
        X_mi,
        y,
        random_state=RANDOM_STATE
    )

    result = pd.DataFrame({
        "feature": X_mi.columns,
        "mutual_information": mi_scores
    })

    result = result.sort_values(
        "mutual_information",
        ascending=False
    )

    print(
        f"\nTop {TOP_MI_FEATURES} "
        f"mutual-information features:"
    )

    print(
        result.head(
            TOP_MI_FEATURES
        ).to_string(index=False)
    )

    return result


# ============================================================
# SAVE ANALYSIS RESULTS
# ============================================================

def save_results(
    name,
    quality,
    variance,
    constant_features,
    near_zero_features,
    duplicate_features,
    correlation_pairs,
    class_difference,
    rf_importance,
    mi_scores
):

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    output_dir = os.path.join(
        OUTPUT_DIR,
        safe_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Variance
    variance_df = pd.DataFrame({
        "feature": variance.index,
        "variance": variance.values
    })

    variance_df = variance_df.sort_values(
        "variance"
    )

    variance_df.to_csv(
        os.path.join(
            output_dir,
            "feature_variance.csv"
        ),
        index=False
    )

    # Constant features
    pd.DataFrame({
        "feature": constant_features
    }).to_csv(
        os.path.join(
            output_dir,
            "constant_features.csv"
        ),
        index=False
    )

    # Near-zero variance
    pd.DataFrame({
        "feature": near_zero_features
    }).to_csv(
        os.path.join(
            output_dir,
            "near_zero_variance_features.csv"
        ),
        index=False
    )

    # Duplicate features
    pd.DataFrame({
        "feature": duplicate_features
    }).to_csv(
        os.path.join(
            output_dir,
            "duplicate_features.csv"
        ),
        index=False
    )

    # Correlation
    correlation_pairs.to_csv(
        os.path.join(
            output_dir,
            "high_correlation_pairs.csv"
        ),
        index=False
    )

    # Class difference
    class_difference.to_csv(
        os.path.join(
            output_dir,
            "class_difference.csv"
        )
    )

    # RF
    rf_importance.to_csv(
        os.path.join(
            output_dir,
            "random_forest_importance.csv"
        ),
        index=False
    )

    # Mutual information
    mi_scores.to_csv(
        os.path.join(
            output_dir,
            "mutual_information.csv"
        ),
        index=False
    )

    # Summary
    summary = pd.DataFrame([
        {
            "samples": quality["samples"],
            "features": quality["features"],
            "missing_values": quality["missing_values"],
            "infinite_values": quality["infinite_values"],
            "constant_features": len(
                constant_features
            ),
            "near_zero_variance_features": len(
                near_zero_features
            ),
            "duplicate_features": len(
                duplicate_features
            ),
            "high_correlation_pairs": len(
                correlation_pairs
            )
        }
    ])

    summary.to_csv(
        os.path.join(
            output_dir,
            "analysis_summary.csv"
        ),
        index=False
    )

    print(
        f"\nResults saved to:\n"
        f"{output_dir}"
    )


# ============================================================
# ANALYZE ONE FEATURE GROUP
# ============================================================

def analyze_group(
    df,
    name
):

    X, y, id_columns, label_column, feature_columns = (
        prepare_features(df)
    )

    quality = analyze_data_quality(
        X,
        name
    )

    variance, constant_features, near_zero_features = (
        analyze_variance(
            X,
            name
        )
    )

    duplicate_features = find_duplicate_features(
        X,
        name
    )

    correlation_pairs_list, correlation_pairs = (
        analyze_correlation(
            X,
            name
        )
    )

    class_difference = analyze_class_difference(
        X,
        y,
        name
    )

    rf_importance = analyze_random_forest(
        X,
        y,
        name
    )

    mi_scores = analyze_mutual_information(
        X,
        y,
        name
    )

    save_results(
        name=name,
        quality=quality,
        variance=variance,
        constant_features=constant_features,
        near_zero_features=near_zero_features,
        duplicate_features=duplicate_features,
        correlation_pairs=correlation_pairs,
        class_difference=class_difference,
        rf_importance=rf_importance,
        mi_scores=mi_scores
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        f"\nInput directory:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    handcrafted = load_feature_file(
        HANDCRAFTED_FILE,
        "Handcrafted"
    )

    cnn = load_feature_file(
        CNN_FILE,
        "CNN"
    )

    metadata = load_feature_file(
        METADATA_FILE,
        "Metadata"
    )

    all_features = load_feature_file(
        ALL_FEATURES_FILE,
        "All Features"
    )

    # --------------------------------------------------------
    # Analyze separately
    # --------------------------------------------------------

    analyze_group(
        handcrafted,
        "Handcrafted"
    )

    analyze_group(
        cnn,
        "CNN"
    )

    analyze_group(
        metadata,
        "Metadata"
    )

    # --------------------------------------------------------
    # Analyze combined feature set
    # --------------------------------------------------------

    analyze_group(
        all_features,
        "All Features"
    )

    print_section(
        "FEATURE ANALYSIS COMPLETED"
    )

    print(
        f"\nAll analysis results are saved in:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nNext stage:"
    )

    print(
        "Feature Selection"
    )


if __name__ == "__main__":
    main()