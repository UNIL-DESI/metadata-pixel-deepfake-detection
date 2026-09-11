from __future__ import annotations

import argparse
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore")


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_required_columns(df):
    required = ["media_id", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def load_data(input_path):
    print_header("LOADING ALIGNED FEATURE DATA")
    print(f"Input file:\n{input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    df = pd.read_csv(input_path)
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


def prepare_features(df):
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

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_count = X.isna().sum().sum()

    if missing_count > 0:
        print(
            f"\nWARNING: {missing_count} missing values found."
        )
        raise ValueError(
            "Missing values detected in feature matrix. "
            "Please handle them before feature selection."
        )

    return X, y, feature_columns


def create_data_split(df, val_size, test_size, random_state):
    print_header("CREATING 70 / 15 / 15 DATA SPLIT")

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=random_state
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


def remove_constant_features(X_train, X_val, X_test):
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


def remove_correlated_features(X_train, X_val, X_test, correlation_threshold):
    print_header("REMOVING HIGHLY CORRELATED FEATURES")
    print(f"Correlation threshold: {correlation_threshold}")
    print(
        f"Calculating correlation matrix for "
        f"{X_train.shape[1]} features..."
    )

    corr_matrix = X_train.corr().abs()

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
            upper[column] > correlation_threshold
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


def calculate_mutual_information(X_train, y_train, random_state):
    print_header("MUTUAL INFORMATION FEATURE RANKING")
    print(
        f"Calculating MI for "
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

    X_values = X_train.values

    mi_scores = mutual_info_classif(
        X_values,
        y_numeric,
        random_state=random_state
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
    print(mi_df.head(20).to_string(index=False))

    return mi_df


def calculate_rf_importance(X_train, y_train, rf_params, random_state):
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
        n_estimators=rf_params.get("n_estimators", 300),
        max_depth=rf_params.get("max_depth", None),
        min_samples_leaf=rf_params.get("min_samples_leaf", 2),
        random_state=random_state,
        n_jobs=rf_params.get("n_jobs", -1),
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
    print(importance_df.head(20).to_string(index=False))

    return importance_df


def combine_rankings(mi_df, rf_df):
    print_header("COMBINING FEATURE RANKINGS")

    ranking_df = mi_df.merge(
        rf_df,
        on="feature",
        how="inner"
    )

    total_features = len(ranking_df)

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


def select_features(ranking_df, n_selected_features):
    print_header("SELECTING FINAL FEATURES")

    available_features = len(ranking_df)
    n_select = min(
        n_selected_features,
        available_features
    )

    selected_features = (
        ranking_df
        .head(n_select)["feature"]
        .tolist()
    )

    print(f"Available features: {available_features}")
    print(f"Requested features: {n_selected_features}")
    print(f"Selected features: {n_select}")

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
        print(f"  {group}: {count}")

    return selected_features


def apply_feature_selection(
    train_df,
    val_df,
    test_df,
    selected_features
):
    print_header("APPLYING SELECTED FEATURES")

    train_selected = pd.concat(
        [
            train_df[["media_id", "label"]].reset_index(drop=True),
            train_df[selected_features].reset_index(drop=True)
        ],
        axis=1
    )

    val_selected = pd.concat(
        [
            val_df[["media_id", "label"]].reset_index(drop=True),
            val_df[selected_features].reset_index(drop=True)
        ],
        axis=1
    )

    test_selected = pd.concat(
        [
            test_df[["media_id", "label"]].reset_index(drop=True),
            test_df[selected_features].reset_index(drop=True)
        ],
        axis=1
    )

    print(f"Training selected shape: {train_selected.shape}")
    print(f"Validation selected shape: {val_selected.shape}")
    print(f"Test selected shape: {test_selected.shape}")

    return train_selected, val_selected, test_selected


def save_results(
    output_dir,
    train_selected,
    val_selected,
    test_selected,
    ranking_df,
    selected_features,
    constant_features,
    correlated_features,
    correlation_threshold
):
    print_header("SAVING FEATURE SELECTION RESULTS")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_selected_160.csv")
    val_path = os.path.join(output_dir, "validation_selected_160.csv")
    test_path = os.path.join(output_dir, "test_selected_160.csv")

    train_selected.to_csv(train_path, index=False)
    val_selected.to_csv(val_path, index=False)
    test_selected.to_csv(test_path, index=False)

    ranking_path = os.path.join(output_dir, "feature_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)

    selected_path = os.path.join(output_dir, "selected_features_160.txt")
    with open(selected_path, "w", encoding="utf-8") as f:
        for i, feature in enumerate(selected_features, start=1):
            f.write(f"{i}. {feature}\n")

    removed_path = os.path.join(output_dir, "removed_features.txt")
    with open(removed_path, "w", encoding="utf-8") as f:
        f.write("CONSTANT FEATURES\n=================\n")
        for feature in constant_features:
            f.write(f"{feature}\n")

        f.write("\n\nHIGHLY CORRELATED FEATURES\n===========================\n")
        for feature in correlated_features:
            f.write(f"{feature}\n")

    report_path = os.path.join(output_dir, "feature_selection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FEATURE SELECTION REPORT\n========================\n\n")
        f.write("Dataset: Deepfake Detection Pilot Dataset\n")
        f.write(f"Total samples: 300\n")
        f.write(f"Train samples: {len(train_selected)}\n")
        f.write(f"Validation samples: {len(val_selected)}\n")
        f.write(f"Test samples: {len(test_selected)}\n\n")
        f.write(f"Constant features removed: {len(constant_features)}\n")
        f.write(f"Highly correlated features removed: {len(correlated_features)}\n")
        f.write(f"Final selected features: {len(selected_features)}\n")

    print("\nFiles created:")
    print(train_path)
    print(val_path)
    print(test_path)
    print(ranking_path)
    print(selected_path)
    print(removed_path)
    print(report_path)


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe feature selection")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config["dataset"]["base_dir"]
    input_path = str(Path(base_dir) / "features" / "aligned" / "aligned_all_features.csv")
    output_dir = str(Path(base_dir) / "features" / "selected")

    random_state = 42
    val_size = 0.15
    test_size = 0.15
    n_selected_features = 160
    correlation_threshold = 0.95
    rf_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1
    }

    print_header("LEAKAGE-SAFE FEATURE SELECTION")
    print(f"\nTarget number of features: {n_selected_features}")

    df = load_data(input_path)

    train_df, val_df, test_df = create_data_split(
        df, val_size, test_size, random_state
    )

    X_train, y_train, feature_columns = prepare_features(train_df)

    X_val = val_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    X_train, X_val, X_test, constant_features = remove_constant_features(
        X_train, X_val, X_test
    )

    X_train, X_val, X_test, correlated_features = remove_correlated_features(
        X_train, X_val, X_test, correlation_threshold
    )

    mi_df = calculate_mutual_information(X_train, y_train, random_state)
    rf_df = calculate_rf_importance(X_train, y_train, rf_params, random_state)

    ranking_df = combine_rankings(mi_df, rf_df)
    selected_features = select_features(ranking_df, n_selected_features)

    train_working = train_df[["media_id", "label"]].copy()
    val_working = val_df[["media_id", "label"]].copy()
    test_working = test_df[["media_id", "label"]].copy()

    for feature in X_train.columns:
        train_working[feature] = X_train[feature].values
        val_working[feature] = X_val[feature].values
        test_working[feature] = X_test[feature].values

    train_selected, val_selected, test_selected = apply_feature_selection(
        train_working,
        val_working,
        test_working,
        selected_features
    )

    save_results(
        output_dir,
        train_selected,
        val_selected,
        test_selected,
        ranking_df,
        selected_features,
        constant_features,
        correlated_features,
        correlation_threshold
    )

    print_header("FEATURE SELECTION COMPLETED")
    print(f"Train samples:      {len(train_selected)}")
    print(f"Validation samples: {len(val_selected)}")
    print(f"Test samples:       {len(test_selected)}")
    print(f"\nFinal feature count: {len(selected_features)}")
    print("\n✓ Feature selection used training data only.")
    print("✓ Validation data was kept separate.")
    print("✓ Test data was kept completely separate.")


if __name__ == "__main__":
    main()