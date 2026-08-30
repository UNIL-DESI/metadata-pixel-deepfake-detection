"""
Feature Selection for Deepfake Detection Pilot Dataset

Purpose:
    Perform leakage-safe feature selection on the 300-video pilot dataset.

Workflow:
    1. Load aligned feature matrix
    2. Stratified 70/15/15 train/validation/test split
    3. Feature selection ONLY on training data
       a. Remove constant features
       b. Remove highly correlated features
       c. Calculate Mutual Information
       d. Calculate Random Forest importance
       e. Combine MI and RF rankings
       f. Select top N features
    4. Apply the selected feature list to train/validation/test
    5. Save datasets and feature-selection report

Important:
    Validation and test data are NEVER used to select features.
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    r"G:\My Drive\DeepfakeResearch\features\aligned"
    r"\aligned_all_features.csv"
)

OUTPUT_DIR = (
    r"G:\My Drive\DeepfakeResearch\features\selected"
)

RANDOM_STATE = 42

# Dataset split
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Number of final selected features
N_SELECTED_FEATURES = 160

# Correlation threshold
CORRELATION_THRESHOLD = 0.95

# Random Forest parameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

warnings.filterwarnings("ignore")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_required_columns(df):
    required = ["media_id", "label"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print_header("LOADING ALIGNED FEATURE DATA")

    print(f"Input file:")
    print(INPUT_PATH)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nDataset shape: {df.shape}")

    check_required_columns(df)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nDataset IDs:")
    print(f"Unique media IDs: {df['media_id'].nunique()}")

    if df["media_id"].duplicated().any():
        raise ValueError(
            "Duplicate media_id values detected."
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Separate identifiers/labels from actual features.
    """

    non_feature_columns = [
        "media_id",
        "label"
    ]

    feature_columns = [
        c for c in df.columns
        if c not in non_feature_columns
    ]

    X = df[feature_columns].copy()
    y = df["label"].copy()

    print(f"\nTotal feature columns: {len(feature_columns)}")

    # Convert all feature columns to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Check missing values
    missing_count = X.isna().sum().sum()

    if missing_count > 0:
        print(
            f"\nWARNING: {missing_count} missing values found."
        )

        # Median imputation using the full feature matrix would technically
        # leak information, so we do not perform it here.
        # Instead, stop and fix the source data.
        raise ValueError(
            "Missing values detected in feature matrix. "
            "Please handle them before feature selection."
        )

    return X, y, feature_columns


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_data_split(df):
    """
    Create stratified 70/15/15 split.

    First:
        70% train
        30% temporary

    Then:
        temporary -> 50% validation + 50% test

    Result:
        70% train
        15% validation
        15% test
    """

    print_header("CREATING 70 / 15 / 15 DATA SPLIT")

    train_df, temp_df = train_test_split(
        df,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    print(
        f"Train:      {len(train_df)} "
        f"({len(train_df) / len(df) * 100:.1f}%)"
    )

    print(
        f"Validation: {len(validation_df)} "
        f"({len(validation_df) / len(df) * 100:.1f}%)"
    )

    print(
        f"Test:       {len(test_df)} "
        f"({len(test_df) / len(df) * 100:.1f}%)"
    )

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation label distribution:")
    print(validation_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    # Verify no overlap
    train_ids = set(train_df["media_id"])
    val_ids = set(validation_df["media_id"])
    test_ids = set(test_df["media_id"])

    if train_ids & val_ids:
        raise ValueError("Train/validation ID overlap detected.")

    if train_ids & test_ids:
        raise ValueError("Train/test ID overlap detected.")

    if val_ids & test_ids:
        raise ValueError("Validation/test ID overlap detected.")

    print("\n✓ No ID overlap between splits.")

    return train_df, validation_df, test_df


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

def remove_constant_features(X_train, X_val, X_test):
    """
    Remove features with exactly zero variance in TRAINING DATA.

    Important:
        The decision is made only using X_train.
    """

    print_header("REMOVING CONSTANT FEATURES")

    variances = X_train.var(axis=0)

    constant_features = variances[
        variances == 0
    ].index.tolist()

    print(
        f"Constant features found: "
        f"{len(constant_features)}"
    )

    if constant_features:
        print("\nRemoved features:")

        for feature in constant_features:
            print(f"  - {feature}")

    keep_features = [
        c for c in X_train.columns
        if c not in constant_features
    ]

    X_train = X_train[keep_features].copy()
    X_val = X_val[keep_features].copy()
    X_test = X_test[keep_features].copy()

    print(
        f"\nFeatures remaining: "
        f"{len(keep_features)}"
    )

    return (
        X_train,
        X_val,
        X_test,
        constant_features
    )


# ============================================================
# REMOVE HIGHLY CORRELATED FEATURES
# ============================================================

def remove_correlated_features(X_train, X_val, X_test):
    """
    Remove redundant features using training data correlation.

    For each pair with |correlation| > threshold,
    one feature is removed.

    IMPORTANT:
        Correlation is calculated ONLY on X_train.
    """

    print_header("REMOVING HIGHLY CORRELATED FEATURES")

    print(
        f"Correlation threshold: "
        f"{CORRELATION_THRESHOLD}"
    )

    print(
        f"Calculating correlation matrix for "
        f"{X_train.shape[1]} features..."
    )

    corr_matrix = X_train.corr().abs()

    # Upper triangular matrix
    upper = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    correlated_features = [
        column
        for column in upper.columns
        if any(
            upper[column] > CORRELATION_THRESHOLD
        )
    ]

    print(
        f"Highly correlated features marked for removal: "
        f"{len(correlated_features)}"
    )

    keep_features = [
        c
        for c in X_train.columns
        if c not in correlated_features
    ]

    X_train = X_train[keep_features].copy()
    X_val = X_val[keep_features].copy()
    X_test = X_test[keep_features].copy()

    print(
        f"Features remaining: "
        f"{len(keep_features)}"
    )

    return (
        X_train,
        X_val,
        X_test,
        correlated_features
    )


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def calculate_mutual_information(X_train, y_train):
    """
    Calculate Mutual Information using TRAINING DATA ONLY.
    """

    print_header("MUTUAL INFORMATION FEATURE RANKING")

    print(
        f"Calculating MI for "
        f"{X_train.shape[1]} features..."
    )

    # Convert labels to integers
    y_numeric = (
        y_train
        .map({
            "real": 0,
            "fake": 1
        })
        .astype(int)
        .values
    )

    X_values = X_train.values

    mi_scores = mutual_info_classif(
        X_values,
        y_numeric,
        random_state=RANDOM_STATE
    )

    mi_df = pd.DataFrame({
        "feature": X_train.columns,
        "mutual_information": mi_scores
    })

    mi_df = mi_df.sort_values(
        "mutual_information",
        ascending=False
    ).reset_index(drop=True)

    print("\nTop 20 MI features:")

    print(
        mi_df.head(20).to_string(index=False)
    )

    return mi_df


# ============================================================
# RANDOM FOREST IMPORTANCE
# ============================================================

def calculate_rf_importance(X_train, y_train):
    """
    Calculate Random Forest feature importance
    using TRAINING DATA ONLY.
    """

    print_header("RANDOM FOREST FEATURE IMPORTANCE")

    print(
        f"Training Random Forest using "
        f"{X_train.shape[1]} features..."
    )

    y_numeric = (
        y_train
        .map({
            "real": 0,
            "fake": 1
        })
        .astype(int)
        .values
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        class_weight="balanced"
    )

    rf.fit(
        X_train,
        y_numeric
    )

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "rf_importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "rf_importance",
        ascending=False
    ).reset_index(drop=True)

    print("\nTop 20 RF features:")

    print(
        importance_df.head(20).to_string(index=False)
    )

    return importance_df


# ============================================================
# COMBINE MI + RANDOM FOREST RANKINGS
# ============================================================

def combine_rankings(mi_df, rf_df):
    """
    Combine Mutual Information and Random Forest rankings.

    We use rank normalization rather than directly averaging
    raw MI and RF values because they are on different scales.

    Final score:

        combined_score =
            0.5 * MI rank score
            +
            0.5 * RF rank score

    Higher score = better feature.
    """

    print_header("COMBINING FEATURE RANKINGS")

    ranking_df = mi_df.merge(
        rf_df,
        on="feature",
        how="inner"
    )

    total_features = len(ranking_df)

    # Rank 1 = best
    ranking_df["mi_rank"] = (
        ranking_df["mutual_information"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    ranking_df["rf_rank"] = (
        ranking_df["rf_importance"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    # Convert ranks into scores between approximately 0 and 1
    ranking_df["mi_rank_score"] = (
        1
        -
        (
            ranking_df["mi_rank"] - 1
        )
        /
        max(total_features - 1, 1)
    )

    ranking_df["rf_rank_score"] = (
        1
        -
        (
            ranking_df["rf_rank"] - 1
        )
        /
        max(total_features - 1, 1)
    )

    ranking_df["combined_score"] = (
        0.5 * ranking_df["mi_rank_score"]
        +
        0.5 * ranking_df["rf_rank_score"]
    )

    ranking_df = ranking_df.sort_values(
        "combined_score",
        ascending=False
    ).reset_index(drop=True)

    ranking_df["final_rank"] = (
        np.arange(len(ranking_df)) + 1
    )

    print("\nTop 30 combined features:")

    print(
        ranking_df[
            [
                "final_rank",
                "feature",
                "mutual_information",
                "rf_importance",
                "combined_score"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    return ranking_df


# ============================================================
# SELECT TOP N FEATURES
# ============================================================

def select_features(ranking_df):
    """
    Select top N features based on combined ranking.
    """

    print_header("SELECTING FINAL FEATURES")

    available_features = len(ranking_df)

    n_select = min(
        N_SELECTED_FEATURES,
        available_features
    )

    selected_features = (
        ranking_df
        .head(n_select)["feature"]
        .tolist()
    )

    print(
        f"Available features: {available_features}"
    )

    print(
        f"Requested features: {N_SELECTED_FEATURES}"
    )

    print(
        f"Selected features: {n_select}"
    )

    print("\nSelected feature groups:")

    counts = {
        "Handcrafted": 0,
        "CNN": 0,
        "Metadata": 0,
        "Other": 0
    }

    for feature in selected_features:

        if feature.startswith("hc_"):
            counts["Handcrafted"] += 1

        elif feature.startswith("cnn_"):
            counts["CNN"] += 1

        elif feature.startswith("meta_"):
            counts["Metadata"] += 1

        else:
            counts["Other"] += 1

    for group, count in counts.items():
        print(
            f"  {group}: {count}"
        )

    return selected_features


# ============================================================
# APPLY FEATURE SELECTION
# ============================================================

def apply_feature_selection(
    train_df,
    val_df,
    test_df,
    selected_features
):
    """
    Apply the SAME selected feature list to all datasets.

    The feature list was created using training data only.
    """

    print_header("APPLYING SELECTED FEATURES")

    train_selected = pd.concat(
        [
            train_df[
                ["media_id", "label"]
            ].reset_index(drop=True),

            train_df[
                selected_features
            ].reset_index(drop=True)
        ],
        axis=1
    )

    val_selected = pd.concat(
        [
            val_df[
                ["media_id", "label"]
            ].reset_index(drop=True),

            val_df[
                selected_features
            ].reset_index(drop=True)
        ],
        axis=1
    )

    test_selected = pd.concat(
        [
            test_df[
                ["media_id", "label"]
            ].reset_index(drop=True),

            test_df[
                selected_features
            ].reset_index(drop=True)
        ],
        axis=1
    )

    print(
        f"Training selected shape: "
        f"{train_selected.shape}"
    )

    print(
        f"Validation selected shape: "
        f"{val_selected.shape}"
    )

    print(
        f"Test selected shape: "
        f"{test_selected.shape}"
    )

    return (
        train_selected,
        val_selected,
        test_selected
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    train_selected,
    val_selected,
    test_selected,
    ranking_df,
    selected_features,
    constant_features,
    correlated_features
):

    print_header("SAVING FEATURE SELECTION RESULTS")

    ensure_output_directory()

    # --------------------------------------------------------
    # Selected datasets
    # --------------------------------------------------------

    train_path = os.path.join(
        OUTPUT_DIR,
        "train_selected_160.csv"
    )

    val_path = os.path.join(
        OUTPUT_DIR,
        "validation_selected_160.csv"
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        "test_selected_160.csv"
    )

    train_selected.to_csv(
        train_path,
        index=False
    )

    val_selected.to_csv(
        val_path,
        index=False
    )

    test_selected.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Feature ranking
    # --------------------------------------------------------

    ranking_path = os.path.join(
        OUTPUT_DIR,
        "feature_ranking.csv"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False
    )

    # --------------------------------------------------------
    # Selected feature names
    # --------------------------------------------------------

    selected_path = os.path.join(
        OUTPUT_DIR,
        "selected_features_160.txt"
    )

    with open(
        selected_path,
        "w",
        encoding="utf-8"
    ) as f:

        for i, feature in enumerate(
            selected_features,
            start=1
        ):
            f.write(
                f"{i}. {feature}\n"
            )

    # --------------------------------------------------------
    # Removed feature list
    # --------------------------------------------------------

    removed_path = os.path.join(
        OUTPUT_DIR,
        "removed_features.txt"
    )

    with open(
        removed_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CONSTANT FEATURES\n"
        )
        f.write(
            "=================\n"
        )

        for feature in constant_features:
            f.write(
                f"{feature}\n"
            )

        f.write(
            "\n\nHIGHLY CORRELATED FEATURES\n"
        )
        f.write(
            "===========================\n"
        )

        for feature in correlated_features:
            f.write(
                f"{feature}\n"
            )

    # --------------------------------------------------------
    # Selection report
    # --------------------------------------------------------

    report_path = os.path.join(
        OUTPUT_DIR,
        "feature_selection_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FEATURE SELECTION REPORT\n"
        )
        f.write(
            "========================\n\n"
        )

        f.write(
            "Dataset: Deepfake Detection Pilot Dataset\n"
        )

        f.write(
            f"Total samples: 300\n"
        )

        f.write(
            f"Train samples: "
            f"{len(train_selected)}\n"
        )

        f.write(
            f"Validation samples: "
            f"{len(val_selected)}\n"
        )

        f.write(
            f"Test samples: "
            f"{len(test_selected)}\n\n"
        )

        f.write(
            "Split ratio:\n"
        )
        f.write(
            "Train = 70%\n"
        )
        f.write(
            "Validation = 15%\n"
        )
        f.write(
            "Test = 15%\n\n"
        )

        f.write(
            "FEATURE SELECTION METHODOLOGY\n"
        )
        f.write(
            "==============================\n\n"
        )

        f.write(
            "1. Feature selection was performed "
            "using training data only.\n"
        )

        f.write(
            "2. Constant features were removed.\n"
        )

        f.write(
            "3. Highly correlated features "
            f"(absolute correlation > "
            f"{CORRELATION_THRESHOLD}) "
            "were removed.\n"
        )

        f.write(
            "4. Mutual Information was calculated "
            "on the training data.\n"
        )

        f.write(
            "5. Random Forest feature importance "
            "was calculated on the training data.\n"
        )

        f.write(
            "6. MI and Random Forest rankings "
            "were combined.\n"
        )

        f.write(
            "7. The top 160 features were selected.\n"
        )

        f.write(
            "8. The same 160 features were applied "
            "to validation and test data.\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Validation and test data were not used "
            "during feature selection.\n\n"
        )

        f.write(
            f"Constant features removed: "
            f"{len(constant_features)}\n"
        )

        f.write(
            f"Highly correlated features removed: "
            f"{len(correlated_features)}\n"
        )

        f.write(
            f"Final selected features: "
            f"{len(selected_features)}\n"
        )

    print("\nFiles created:")

    print(train_path)
    print(val_path)
    print(test_path)
    print(ranking_path)
    print(selected_path)
    print(removed_path)
    print(report_path)

    return {
        "train": train_path,
        "validation": val_path,
        "test": test_path,
        "ranking": ranking_path,
        "selected_features": selected_path,
        "removed_features": removed_path,
        "report": report_path
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "LEAKAGE-SAFE FEATURE SELECTION"
    )

    print(
        f"\nTarget number of features: "
        f"{N_SELECTED_FEATURES}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Feature selection will be performed "
        "ONLY on the training set."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    train_df, val_df, test_df = (
        create_data_split(df)
    )

    # --------------------------------------------------------
    # Prepare feature matrices
    # --------------------------------------------------------

    X_train, y_train, feature_columns = (
        prepare_features(train_df)
    )

    X_val = val_df[
        feature_columns
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Constant feature removal
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        constant_features
    ) = remove_constant_features(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # Correlation filtering
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        correlated_features
    ) = remove_correlated_features(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # Mutual Information
    # --------------------------------------------------------

    mi_df = calculate_mutual_information(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf_df = calculate_rf_importance(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Combine rankings
    # --------------------------------------------------------

    ranking_df = combine_rankings(
        mi_df,
        rf_df
    )

    # --------------------------------------------------------
    # Select final 160
    # --------------------------------------------------------

    selected_features = select_features(
        ranking_df
    )

    # --------------------------------------------------------
    # Apply selected features
    # --------------------------------------------------------

    train_working = train_df[
        ["media_id", "label"]
    ].copy()

    val_working = val_df[
        ["media_id", "label"]
    ].copy()

    test_working = test_df[
        ["media_id", "label"]
    ].copy()

    # Add feature matrices
    for feature in X_train.columns:

        train_working[feature] = (
            X_train[feature].values
        )

        val_working[feature] = (
            X_val[feature].values
        )

        test_working[feature] = (
            X_test[feature].values
        )

    (
        train_selected,
        val_selected,
        test_selected
    ) = apply_feature_selection(
        train_working,
        val_working,
        test_working,
        selected_features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        train_selected,
        val_selected,
        test_selected,
        ranking_df,
        selected_features,
        constant_features,
        correlated_features
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print_header(
        "FEATURE SELECTION COMPLETED"
    )

    print(
        f"Train samples:      "
        f"{len(train_selected)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_selected)}"
    )

    print(
        f"Test samples:       "
        f"{len(test_selected)}"
    )

    print(
        f"\nFinal feature count: "
        f"{len(selected_features)}"
    )

    print(
        "\n✓ Feature selection used training data only."
    )

    print(
        "✓ Validation data was kept separate."
    )

    print(
        "✓ Test data was kept completely separate."
    )

    print(
        "\nNext stage:"
    )

    print(
        "Model training and evaluation."
    )


if __name__ == "__main__":
    main()