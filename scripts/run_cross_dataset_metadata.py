import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


ALIGNED_METADATA = "features/aligned/aligned_metadata.csv"
INVENTORY = "dataset_inventory.csv"


def prepare_data():
    meta = pd.read_csv(ALIGNED_METADATA)
    inventory = pd.read_csv(INVENTORY)

    if "media_id" not in meta.columns:
        raise ValueError("aligned_metadata.csv must contain media_id")

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

    return df


def run_experiment(df, train_datasets, test_dataset):

    print("\n" + "=" * 80)
    print(
        f"TRAIN: {train_datasets}  ->  TEST: {test_dataset}"
    )
    print("=" * 80)

    train_df = df[df["dataset"].isin(train_datasets)].copy()
    test_df = df[df["dataset"] == test_dataset].copy()

    print("\nTrain samples:", len(train_df))
    print("Test samples:", len(test_df))

    print("\nTrain labels:")
    print(train_df["label"].value_counts())

    print("\nTest labels:")
    print(test_df["label"].value_counts())

    feature_cols = [
        c for c in df.columns
        if c not in [
            "media_id",
            "label",
            "dataset"
        ]
    ]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df["label"].map({
        "real": 0,
        "fake": 1
    })

    y_test = test_df["label"].map({
        "real": 0,
        "fake": 1
    })

    # -----------------------------------------------------
    # Remove constants based ONLY on training data
    # -----------------------------------------------------
    valid_features = [
        c for c in feature_cols
        if X_train[c].nunique() > 1
    ]

    X_train = X_train[valid_features]
    X_test = X_test[valid_features]

    print("\nFeatures used:", len(valid_features))

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, prob)
    except ValueError:
        auc = np.nan

    print("\nRESULTS")
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            pred,
            target_names=["real", "fake"],
            zero_division=0
        )
    )

    return {
        "train_datasets": "+".join(train_datasets),
        "test_dataset": test_dataset,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "num_features": len(valid_features)
    }


def main():

    df = prepare_data()

    print("=" * 80)
    print("CROSS-DATASET METADATA GENERALIZATION")
    print("=" * 80)

    print("\nTotal samples:", len(df))

    print("\nDataset distribution:")
    print(
        df.groupby(
            ["dataset", "label"]
        ).size()
    )

    experiments = [
        (["FF++", "Celeb-DF"], "DFDC"),
        (["FF++", "DFDC"], "Celeb-DF"),
        (["Celeb-DF", "DFDC"], "FF++"),
    ]

    results = []

    for train_datasets, test_dataset in experiments:
        result = run_experiment(
            df,
            train_datasets,
            test_dataset
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("FINAL CROSS-DATASET SUMMARY")
    print("=" * 80)

    print(results_df.to_string(index=False))

    results_df.to_csv(
        "results/cross_dataset_metadata_results.csv",
        index=False
    )

    print(
        "\nSaved to:"
        "\nresults/cross_dataset_metadata_results.csv"
    )


if __name__ == "__main__":
    main()