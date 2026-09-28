import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALIGNED_METADATA = "features/aligned/aligned_metadata.csv"
INVENTORY = "dataset_inventory.csv"
OUTPUT = "results/metadata_bias_screening.csv"


def safe_auc_binary(X, y):
    """
    Measure how well ONE feature predicts real/fake.
    Returns ROC-AUC, made symmetric so 0.2 becomes 0.8 signal strength.
    """
    try:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])

        model.fit(X, y)

        prob = model.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, prob)

        # Direction doesn't matter for signal strength
        return max(auc, 1 - auc)

    except Exception:
        return np.nan


def dataset_origin_score_single_feature(X, y_dataset):
    """
    Measure how predictive one feature is of dataset origin.
    Uses training accuracy on this diagnostic only.
    """
    try:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            ))
        ])

        model.fit(X, y_dataset)

        return model.score(X, y_dataset)

    except Exception:
        return np.nan


def main():

    print("=" * 80)
    print("METADATA DATASET-BIAS FEATURE SCREENING")
    print("=" * 80)

    meta = pd.read_csv(ALIGNED_METADATA)
    inventory = pd.read_csv(INVENTORY)

    print("Metadata shape:", meta.shape)
    print("Inventory shape:", inventory.shape)

    # ---------------------------------------------------------
    # Prepare IDs
    # ---------------------------------------------------------
    inventory["media_id"] = (
        inventory["video_name"]
        .astype(str)
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    df = meta.merge(
        inventory[["media_id", "dataset"]],
        on="media_id",
        how="inner"
    )

    print("Merged shape:", df.shape)

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------
    y_label = df["label"].map({
        "real": 0,
        "fake": 1
    })

    dataset_encoder = LabelEncoder()

    y_dataset = dataset_encoder.fit_transform(
        df["dataset"]
    )

    print("\nDataset classes:")
    print(dataset_encoder.classes_)

    # ---------------------------------------------------------
    # Feature columns
    # ---------------------------------------------------------
    exclude = [
        "media_id",
        "label",
        "dataset"
    ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude
    ]

    # Remove constants
    feature_cols = [
        c for c in feature_cols
        if df[c].nunique() > 1
    ]

    print("\nFeatures evaluated:", len(feature_cols))

    results = []

    # ---------------------------------------------------------
    # Evaluate each feature independently
    # ---------------------------------------------------------
    for feature in feature_cols:

        X = df[[feature]].copy()

        if not np.issubdtype(X[feature].dtype, np.number):
            X = pd.get_dummies(
                X,
                columns=[feature],
                drop_first=False
            )

        # -----------------------------------------------------
        # Mutual information with real/fake
        # -----------------------------------------------------
        try:
            mi_label = mutual_info_classif(
                X,
                y_label,
                random_state=42
            ).mean()
        except Exception:
            mi_label = np.nan

        # -----------------------------------------------------
        # Mutual information with dataset origin
        # -----------------------------------------------------
        try:
            mi_dataset = mutual_info_classif(
                X,
                y_dataset,
                random_state=42
            ).mean()
        except Exception:
            mi_dataset = np.nan

        # -----------------------------------------------------
        # Single-feature binary prediction
        # -----------------------------------------------------
        label_auc = safe_auc_binary(
            X,
            y_label
        )

        # -----------------------------------------------------
        # Single-feature dataset prediction
        # -----------------------------------------------------
        dataset_accuracy = dataset_origin_score_single_feature(
            X,
            y_dataset
        )

        # -----------------------------------------------------
        # Bias-aware score
        # -----------------------------------------------------
        mi_ratio = (
            mi_label / (mi_dataset + 1e-8)
            if not np.isnan(mi_label)
            and not np.isnan(mi_dataset)
            else np.nan
        )

        bias_score = (
            mi_label - mi_dataset
            if not np.isnan(mi_label)
            and not np.isnan(mi_dataset)
            else np.nan
        )

        results.append({
            "feature": feature,
            "label_mutual_information": mi_label,
            "dataset_mutual_information": mi_dataset,
            "label_auc": label_auc,
            "dataset_origin_accuracy": dataset_accuracy,
            "label_to_dataset_mi_ratio": mi_ratio,
            "bias_aware_score": bias_score
        })

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Rankings
    # ---------------------------------------------------------
    results_df = results_df.sort_values(
        by=[
            "bias_aware_score",
            "label_mutual_information"
        ],
        ascending=False
    )

    print("\n" + "=" * 80)
    print("BIAS SCREENING RESULTS")
    print("=" * 80)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(results_df.to_string(index=False))

    # ---------------------------------------------------------
    # Recommended candidates
    # ---------------------------------------------------------
    candidates = results_df[
        (
            results_df["label_mutual_information"] > 0.01
        )
        &
        (
            results_df["dataset_origin_accuracy"] < 0.70
        )
    ].copy()

    print("\n" + "=" * 80)
    print("LOWER-BIAS METADATA CANDIDATES")
    print("=" * 80)

    if len(candidates) == 0:
        print(
            "No metadata features passed the current thresholds."
        )
    else:
        print(
            candidates.to_string(index=False)
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    results_df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()