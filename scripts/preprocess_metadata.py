from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================
# LOAD DATA
# ============================================================

def load_metadata(input_file):
    print("\nLoading metadata...")
    df = pd.read_csv(input_file, dtype={"media_id": str})
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    return df


# ============================================================
# CLEAN MISSING VALUES
# ============================================================

def clean_missing_values(df):
    print("\nCleaning missing values...")
    df = df.copy()

    missing_values = [
        "Unknown",
        "unknown",
        "UNKNOWN",
        "",
        "None",
        "none",
        "N/A",
        "n/a",
        "NA",
        "na"
    ]

    df = df.replace(missing_values, np.nan)
    df = df.replace(-1, np.nan)

    return df


# ============================================================
# SEPARATE TARGET AND IDENTIFIERS
# ============================================================

def separate_target(df):
    print("\nSeparating target and identifiers...")

    y = df["label"].copy()
    y = y.map({
        "real": 0,
        "fake": 1
    })

    if y.isna().any():
        invalid_labels = df.loc[
            y.isna(),
            "label"
        ].unique()
        raise ValueError(
            f"Unexpected labels found: {invalid_labels}"
        )

    identifier_columns = [
        "label",
        "media_id"
    ]

    if "dataset" in df.columns:
        identifier_columns.append("dataset")

    X = df.drop(
        columns=identifier_columns,
        errors="ignore"
    )

    return X, y


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(X):
    print("\nIdentifying feature types...")

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features:")
    for feature in numerical_features:
        print(f"  - {feature}")

    print("\nCategorical features:")
    for feature in categorical_features:
        print(f"  - {feature}")

    return numerical_features, categorical_features


# ============================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(
    numerical_features,
    categorical_features
):
    print("\nBuilding preprocessing pipeline...")

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# GENERATE FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor):
    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    cleaned_names = []
    for name in feature_names:
        name = name.replace(
            "numerical__",
            ""
        )
        name = name.replace(
            "categorical__",
            ""
        )
        cleaned_names.append(name)

    return cleaned_names


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess metadata features")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_config = config["features"]["metadata"]
    input_file = metadata_config["output"]

    output_dir = str(Path(input_file).parent / "metadata_preprocessed")

    output_features = os.path.join(
        output_dir,
        "metadata_feature_matrix.csv"
    )

    output_labels = os.path.join(
        output_dir,
        "metadata_labels.csv"
    )

    output_pipeline = os.path.join(
        output_dir,
        "metadata_preprocessing_info.txt"
    )

    print("=" * 60)
    print("METADATA PREPROCESSING")
    print("=" * 60)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    df = load_metadata(input_file)
    df = clean_missing_values(df)
    if "media_id" not in df.columns:
        raise ValueError("Raw metadata does not contain media_id.")
    if df["media_id"].isna().any():
        raise ValueError("Raw metadata contains missing media_id values.")
    df["media_id"] = df["media_id"].astype(str).str.strip()
    if df["media_id"].duplicated().any():
        raise ValueError("Raw metadata contains duplicate media_id values.")
    X, y = separate_target(df)

    print("\nTarget distribution:")
    print(
        y.value_counts()
        .rename({
            0: "real",
            1: "fake"
        })
    )

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    # Data splits must already exist and must match raw metadata IDs.
    # The training split alone determines preprocessing statistics and categories.
    split_dir = Path("data/splits")
    split_files = {
        "train": split_dir / "train_ids.csv",
        "validation": split_dir / "validation_ids.csv",
        "test": split_dir / "test_ids.csv",
    }
    split_ids = {}
    for name, split_path in split_files.items():
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing {name} split: {split_path}. "
                "Run scripts/run_split_dataset.py first."
            )
        ids_df = pd.read_csv(split_path, dtype={"media_id": str})
        if "media_id" not in ids_df.columns or ids_df["media_id"].isna().any():
            raise ValueError(f"Invalid split IDs in {split_path}")
        ids = ids_df["media_id"].str.strip()
        if ids.duplicated().any():
            raise ValueError(f"Duplicate IDs in {split_path}")
        split_ids[name] = set(ids)

    if (split_ids["train"] & split_ids["validation"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["validation"] & split_ids["test"]):
        raise ValueError("Overlapping IDs between dataset splits.")
    union_ids = set().union(*split_ids.values())
    actual_ids = set(df["media_id"])
    if union_ids != actual_ids:
        raise ValueError(
            "Split IDs and metadata IDs differ: "
            f"missing from metadata={sorted(union_ids-actual_ids)[:10]}, "
            f"missing from splits={sorted(actual_ids-union_ids)[:10]}"
        )
    train_mask = df["media_id"].isin(split_ids["train"])
    if not train_mask.any():
        raise ValueError("No training metadata rows found.")

    print("\nFitting metadata preprocessor on TRAINING rows only...")
    preprocessor.fit(X.loc[train_mask])
    print("Transforming every row using training-fitted parameters...")
    X_processed = preprocessor.transform(X)
    joblib.dump(preprocessor, Path(output_dir) / "metadata_preprocessor.joblib")

    print(
        f"Original feature count: {X.shape[1]}"
    )
    print(
        f"Processed feature count: {X_processed.shape[1]}"
    )

    feature_names = get_feature_names(
        preprocessor
    )

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    if "media_id" in df.columns:
        X_processed_df.insert(
            0,
            "media_id",
            df["media_id"].values
        )

    X_processed_df.to_csv(
        output_features,
        index=False
    )

    labels_df = pd.DataFrame({
        "label": y.values
    })

    if "media_id" in df.columns:
        labels_df.insert(
            0,
            "media_id",
            df["media_id"].values
        )

    labels_df.to_csv(
        output_labels,
        index=False
    )

    with open(
        output_pipeline,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "METADATA PREPROCESSING INFORMATION (TRAIN-FITTED)\n"
        )
        f.write("=" * 50 + "\n\n")
        f.write(
            f"Original samples: {len(df)}\n"
        )
        f.write(
            f"Original features: {X.shape[1]}\n"
        )
        f.write(
            f"Processed features: "
            f"{X_processed.shape[1]}\n\n"
        )
        f.write(
            "Numerical features:\n"
        )
        for feature in numerical_features:
            f.write(
                f"  {feature}\n"
            )
        f.write(
            "\nCategorical features:\n"
        )
        for feature in categorical_features:
            f.write(
                f"  {feature}\n"
            )
        f.write(
            "\nProcessing:\n"
        )
        f.write(
            "Numerical missing values -> median\n"
        )
        f.write(
            "Numerical features -> StandardScaler\n"
        )
        f.write(
            "Categorical missing values -> most frequent\n"
        )
        f.write(
            "Categorical features -> OneHotEncoder\n"
        )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"\nFeature matrix saved to:\n"
        f"{output_features}"
    )

    print(
        f"\nLabels saved to:\n"
        f"{output_labels}"
    )

    print(
        f"\nPreprocessing information saved to:\n"
        f"{output_pipeline}"
    )

    print(
        f"\nFinal feature matrix shape: "
        f"{X_processed_df.shape}"
    )


if __name__ == "__main__":
    main()