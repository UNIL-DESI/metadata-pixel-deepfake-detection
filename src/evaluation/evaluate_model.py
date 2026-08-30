"""Evaluate a saved baseline model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_saved_model(model_path: str | Path, features_csv: str | Path) -> dict:
    """Evaluate a saved model on a feature CSV."""
    bundle = joblib.load(model_path)

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(features_csv).fillna(0)
    X = df[feature_cols]
    y = label_encoder.transform(df["label"])

    preds = model.predict(X)

    results = {
        "accuracy": accuracy_score(y, preds),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(
            y,
            preds,
            target_names=label_encoder.classes_,
            output_dict=True,
        ),
    }

    print("Accuracy:", results["accuracy"])
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds, target_names=label_encoder.classes_))

    if hasattr(model, "predict_proba") and len(label_encoder.classes_) == 2:
        probs = model.predict_proba(X)[:, 1]
        results["roc_auc"] = roc_auc_score(y, probs)
        print("ROC-AUC:", results["roc_auc"])

    return results
