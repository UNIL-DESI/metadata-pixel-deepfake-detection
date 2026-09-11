from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RANDOM_STATE = 42
VARIANCE_THRESHOLD = 1e-8
CORRELATION_THRESHOLD = 0.95
TOP_RF_FEATURES = 50
TOP_MI_FEATURES = 50


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_feature_file(path, name):
    print(f"\nLoading {name}...")
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:\n{path}")
    df = pd.read_csv(path)
    print(f"{name} shape: {df.shape}")
    return df


def identify_columns(df):
    identifier_candidates = ["media_id", "video_id", "video_name"]
    id_columns = [col for col in identifier_candidates if col in df.columns]
    label_column = "label" if "label" in df.columns else None
    feature_columns = [col for col in df.columns if col not in id_columns and col != label_column]
    return id_columns, label_column, feature_columns


def convert_labels(y):
    y = y.astype(str).str.lower().str.strip()
    mapping = {"real": 0, "fake": 1, "0": 0, "1": 1}
    converted = y.map(mapping)
    if converted.isna().any():
        invalid = sorted(y[converted.isna()].unique())
        raise ValueError(f"Unexpected labels found: {invalid}")
    return converted.astype(int)


def prepare_features(df):
    id_columns, label_column, feature_columns = identify_columns(df)
    if label_column is None:
        raise ValueError("Dataset does not contain a 'label' column.")
    X = df[feature_columns].copy()
    y = convert_labels(df[label_column])
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y, id_columns, label_column, feature_columns


def analyze_data_quality(X, name):
    print_section(f"{name} - DATA QUALITY")
    results = {
        "samples": len(X),
        "features": X.shape[1],
        "missing_values": int(X.isna().sum().sum()),
        "infinite_values": int(np.isinf(X.select_dtypes(include=np.number)).sum().sum())
    }
    return results


def analyze_variance(X, name):
    print_section(f"{name} - VARIANCE ANALYSIS")
    variance = X.var(axis=0, skipna=True)
    constant_features = variance[variance == 0].index.tolist()
    near_zero_features = variance[variance <= VARIANCE_THRESHOLD].index.tolist()
    return variance, constant_features, near_zero_features


def find_duplicate_features(X, name):
    print_section(f"{name} - DUPLICATE FEATURE ANALYSIS")
    duplicate_mask = X.T.duplicated()
    duplicates = X.columns[duplicate_mask].tolist()
    return duplicates


def analyze_correlation(X, name):
    print_section(f"{name} - CORRELATION ANALYSIS")
    if X.shape[1] == 0:
        return [], pd.DataFrame()
    correlation = X.corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    highly_correlated = []
    for column in upper.columns:
        correlated_columns = upper.index[upper[column] > CORRELATION_THRESHOLD].tolist()
        for row in correlated_columns:
            highly_correlated.append({
                "feature_1": row,
                "feature_2": column,
                "correlation": upper.loc[row, column]
            })
    return highly_correlated, pd.DataFrame(highly_correlated)


def analyze_class_difference(X, y, name):
    print_section(f"{name} - CLASS-WISE ANALYSIS")
    real_mean = X.loc[y == 0].mean()
    fake_mean = X.loc[y == 1].mean()
    difference = (fake_mean - real_mean).abs()
    result = pd.DataFrame({
        "real_mean": real_mean,
        "fake_mean": fake_mean,
        "absolute_difference": difference
    }).sort_values("absolute_difference", ascending=False)
    return result


def analyze_random_forest(X, y, name):
    print_section(f"{name} - PRELIMINARY RANDOM FOREST IMPORTANCE")
    X_rf = X.copy().replace([np.inf, -np.inf], np.nan).fillna(X.median())
    variance = X_rf.var()
    X_rf = X_rf[variance[variance > VARIANCE_THRESHOLD].index]

    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_rf, y)
    importance = pd.DataFrame({
        "feature": X_rf.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return importance


def analyze_mutual_information(X, y, name):
    print_section(f"{name} - MUTUAL INFORMATION")
    X_mi = X.copy().replace([np.inf, -np.inf], np.nan).fillna(X.median())
    variance = X_mi.var()
    X_mi = X_mi[variance[variance > VARIANCE_THRESHOLD].index]

    mi_scores = mutual_info_classif(X_mi, y, random_state=RANDOM_STATE)
    result = pd.DataFrame({
        "feature": X_mi.columns,
        "mutual_information": mi_scores
    }).sort_values("mutual_information", ascending=False)
    return result


def save_results(name, output_dir_base, quality, variance, constant_features, near_zero_features, duplicate_features, correlation_pairs, class_difference, rf_importance, mi_scores):
    safe_name = name.lower().replace(" ", "_").replace("+", "plus")
    output_dir = os.path.join(output_dir_base, safe_name)
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({"feature": variance.index, "variance": variance.values}).sort_values("variance").to_csv(os.path.join(output_dir, "feature_variance.csv"), index=False)
    pd.DataFrame({"feature": constant_features}).to_csv(os.path.join(output_dir, "constant_features.csv"), index=False)
    pd.DataFrame({"feature": near_zero_features}).to_csv(os.path.join(output_dir, "near_zero_variance_features.csv"), index=False)
    pd.DataFrame({"feature": duplicate_features}).to_csv(os.path.join(output_dir, "duplicate_features.csv"), index=False)
    correlation_pairs.to_csv(os.path.join(output_dir, "high_correlation_pairs.csv"), index=False)
    class_difference.to_csv(os.path.join(output_dir, "class_difference.csv"))
    rf_importance.to_csv(os.path.join(output_dir, "random_forest_importance.csv"), index=False)
    mi_scores.to_csv(os.path.join(output_dir, "mutual_information.csv"), index=False)


def analyze_group(df, name, output_dir):
    X, y, _, _, _ = prepare_features(df)
    quality = analyze_data_quality(X, name)
    variance, constant_features, near_zero_features = analyze_variance(X, name)
    duplicate_features = find_duplicate_features(X, name)
    _, correlation_pairs = analyze_correlation(X, name)
    class_difference = analyze_class_difference(X, y, name)
    rf_importance = analyze_random_forest(X, y, name)
    mi_scores = analyze_mutual_information(X, y, name)

    save_results(
        name, output_dir, quality, variance, constant_features,
        near_zero_features, duplicate_features, correlation_pairs,
        class_difference, rf_importance, mi_scores
    )


def main():
    parser = argparse.ArgumentParser(description="Run feature analysis")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config["dataset"]["base_dir"]
    input_dir = str(Path(base_dir) / "features" / "aligned")
    output_dir = str(Path(base_dir) / "features" / "analysis")

    os.makedirs(output_dir, exist_ok=True)

    handcrafted_file = os.path.join(input_dir, "aligned_handcrafted.csv")
    cnn_file = os.path.join(input_dir, "aligned_cnn.csv")
    metadata_file = os.path.join(input_dir, "aligned_metadata.csv")
    all_features_file = os.path.join(input_dir, "aligned_all_features.csv")

    handcrafted = load_feature_file(handcrafted_file, "Handcrafted")
    cnn = load_feature_file(cnn_file, "CNN")
    metadata = load_feature_file(metadata_file, "Metadata")
    all_features = load_feature_file(all_features_file, "All Features")

    analyze_group(handcrafted, "Handcrafted", output_dir)
    analyze_group(cnn, "CNN", output_dir)
    analyze_group(metadata, "Metadata", output_dir)
    analyze_group(all_features, "All Features", output_dir)

    print_section("FEATURE ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()