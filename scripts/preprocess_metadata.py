import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "G:/My Drive/DeepfakeResearch/features/metadata/metadata_features.csv"

OUTPUT_DIR = "G:/My Drive/DeepfakeResearch/features/metadata/metadata_preprocessed"

OUTPUT_FEATURES = os.path.join(
    OUTPUT_DIR,
    "metadata_feature_matrix.csv"
)

OUTPUT_LABELS = os.path.join(
    OUTPUT_DIR,
    "metadata_labels.csv"
)

OUTPUT_PIPELINE = os.path.join(
    OUTPUT_DIR,
    "metadata_preprocessing_info.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_metadata():

    print("\nLoading metadata...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# CLEAN MISSING VALUES
# ============================================================

def clean_missing_values(df):

    print("\nCleaning missing values...")

    df = df.copy()

    # --------------------------------------------------------
    # Convert common missing-value representations to NaN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # -1 is used in your dataset for unavailable metadata
    # --------------------------------------------------------

    df = df.replace(-1, np.nan)

    return df


# ============================================================
# SEPARATE TARGET AND IDENTIFIERS
# ============================================================

def separate_target(df):

    print("\nSeparating target and identifiers...")

    # Target
    y = df["label"].copy()

    # Convert labels to numeric
    y = y.map({
        "real": 0,
        "fake": 1
    })

    # Check for unexpected labels
    if y.isna().any():

        invalid_labels = df.loc[
            y.isna(),
            "label"
        ].unique()

        raise ValueError(
            f"Unexpected labels found: {invalid_labels}"
        )

    # --------------------------------------------------------
    # Metadata identifiers that should NOT be used as features
    # --------------------------------------------------------

    identifier_columns = [
        "label",
        "media_id"
    ]

    # Keep dataset separately for analysis,
    # but don't give it directly to the classifier.
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

    # --------------------------------------------------------
    # Numerical preprocessing
    #
    # Missing values -> median
    # Scaling -> StandardScaler
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Categorical preprocessing
    #
    # Missing values -> most frequent value
    # Encoding -> One-Hot Encoding
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Combine both
    # --------------------------------------------------------

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

    # Remove sklearn prefixes
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

    print("=" * 60)
    print("METADATA PREPROCESSING")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_metadata()

    # --------------------------------------------------------
    # 2. Clean missing values
    # --------------------------------------------------------

    df = clean_missing_values(df)

    # --------------------------------------------------------
    # 3. Separate target
    # --------------------------------------------------------

    X, y = separate_target(df)

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .rename({
            0: "real",
            1: "fake"
        })
    )

    # --------------------------------------------------------
    # 4. Identify feature types
    # --------------------------------------------------------

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    # --------------------------------------------------------
    # 5. Build preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    # --------------------------------------------------------
    # 6. Fit and transform
    # --------------------------------------------------------

    print("\nApplying preprocessing...")

    X_processed = preprocessor.fit_transform(X)

    print(
        f"Original feature count: {X.shape[1]}"
    )

    print(
        f"Processed feature count: {X_processed.shape[1]}"
    )

    # --------------------------------------------------------
    # 7. Get feature names
    # --------------------------------------------------------

    feature_names = get_feature_names(
        preprocessor
    )

    # --------------------------------------------------------
    # 8. Create feature matrix
    # --------------------------------------------------------

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    # Preserve media_id for later traceability
    if "media_id" in df.columns:

        X_processed_df.insert(
            0,
            "media_id",
            df["media_id"].values
        )

    # --------------------------------------------------------
    # 9. Save feature matrix
    # --------------------------------------------------------

    X_processed_df.to_csv(
        OUTPUT_FEATURES,
        index=False
    )

    # --------------------------------------------------------
    # 10. Save labels
    # --------------------------------------------------------

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
        OUTPUT_LABELS,
        index=False
    )

    # --------------------------------------------------------
    # 11. Save preprocessing information
    # --------------------------------------------------------

    with open(
        OUTPUT_PIPELINE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "METADATA PREPROCESSING INFORMATION\n"
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

    # --------------------------------------------------------
    # 12. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"\nFeature matrix saved to:\n"
        f"{OUTPUT_FEATURES}"
    )

    print(
        f"\nLabels saved to:\n"
        f"{OUTPUT_LABELS}"
    )

    print(
        f"\nPreprocessing information saved to:\n"
        f"{OUTPUT_PIPELINE}"
    )

    print(
        f"\nFinal feature matrix shape: "
        f"{X_processed_df.shape}"
    )


if __name__ == "__main__":
    main()