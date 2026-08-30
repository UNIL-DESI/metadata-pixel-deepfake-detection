"""Train a simple baseline classifier on feature CSV files."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ID_COLUMNS = {
    "label",
    "video_name",
    "frame_name",
    "dataset",
    "filepath",
    "video_path",
    "split",
}


def load_xy(features_csv: str | Path):
    """Load numeric features and labels from a CSV."""
    df = pd.read_csv(features_csv)
    df = df.fillna(0)

    y_raw = df["label"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    feature_cols = [
        col
        for col in df.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[feature_cols]

    return df, X, y, label_encoder, feature_cols


def train_random_forest(
    features_csv: str | Path,
    model_output: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train a Random Forest baseline using video-level features."""
    df, X, y, label_encoder, feature_cols = load_xy(features_csv)

    stratify = y if len(set(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    if hasattr(model, "predict_proba") and len(label_encoder.classes_) == 2:
        probs = model.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", roc_auc_score(y_test, probs))

    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
        },
        model_output,
    )

    print(f"Saved model: {model_output}")
    return model
