import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ALIGNED_METADATA = "features/aligned/aligned_metadata.csv"
INVENTORY = "dataset_inventory.csv"


def main():
    print("=" * 70)
    print("DATASET-ORIGIN PREDICTION USING METADATA")
    print("=" * 70)

    meta = pd.read_csv(ALIGNED_METADATA)
    inventory = pd.read_csv(INVENTORY)

    print("Metadata shape:", meta.shape)
    print("Inventory shape:", inventory.shape)

    # ---------------------------------------------------------
    # Standardize identifier name
    # ---------------------------------------------------------
    if "media_id" not in meta.columns:
        raise ValueError("aligned_metadata.csv must contain media_id")

    # Build media_id in inventory from video_name
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

    print("\nMerged shape:", df.shape)

    print("\nDataset distribution:")
    print(df["dataset"].value_counts())

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------
    drop_cols = [
        c for c in [
            "media_id",
            "label",
            "dataset"
        ]
        if c in df.columns
    ]

    X = df.drop(columns=drop_cols)
    y = df["dataset"]

    # Remove constant features
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    if constant_cols:
        print("\nRemoving constant features:")
        for c in constant_cols:
            print("-", c)

        X = X.drop(columns=constant_cols)

    print("\nFeatures used:", X.shape[1])

    # ---------------------------------------------------------
    # Stratified split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        pred,
        labels=["FF++", "Celeb-DF", "DFDC"]
    ))

    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    # Chance level = 1 / 3
    print("\nChance-level accuracy: ~0.3333")

    if accuracy > 0.80:
        print(
            "\nWARNING: Metadata strongly predicts dataset origin."
        )
    elif accuracy > 0.60:
        print(
            "\nMetadata contains noticeable dataset-origin information."
        )
    else:
        print(
            "\nDataset-origin information appears relatively limited."
        )


if __name__ == "__main__":
    main()