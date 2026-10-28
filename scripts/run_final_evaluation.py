from __future__ import annotations

"""Refit the highest-validation-F1 pipeline and evaluate on untouched TEST."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from run_train_models import (
    convert_labels, load_configuration, load_split,
    resolve_directories, split_fingerprint, validate_splits,
)


def read_selected_features(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature list: {path}")
    features = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        prefix, separator, rest = line.partition(". ")
        features.append(rest if separator and prefix.isdigit() else line)
    if not features or len(features) != len(set(features)):
        raise ValueError(f"Empty or duplicate feature names in {path}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected model on held-out test set")
    parser.add_argument("--config", default="full.yaml")
    args = parser.parse_args()
    config = load_configuration(args.config)
    selected_dir, model_dir, result_dir = resolve_directories(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_info_path = result_dir / "best_model_validation.json"
    if not best_info_path.is_file():
        raise FileNotFoundError(
            f"Missing validation winner: {best_info_path}. Run run_train_models.py first."
        )
    best_info = json.loads(best_info_path.read_text(encoding="utf-8"))
    best_name = best_info.get("best_model")
    if best_info.get("selection_metric") != "validation_f1_score":
        raise ValueError("Best-model metadata must select by validation_f1_score")
    if best_info.get("test_used") is not False:
        raise ValueError("Best-model selection record indicates the test set was used")
    if best_name not in {"logistic_regression", "random_forest", "svm"}:
        raise ValueError(f"Unsupported selected model: {best_name}")

    train, features = load_split(selected_dir / "train_selected_160.csv", "train")
    validation, val_features = load_split(selected_dir / "validation_selected_160.csv", "validation")
    test, test_features = load_split(selected_dir / "test_selected_160.csv", "test")
    validate_splits(train, validation, test, [features, val_features, test_features])
    selected_names = read_selected_features(selected_dir / "selected_features_160.txt")
    if features != selected_names:
        raise ValueError("Train CSV feature order differs from saved feature-selection list")
    if features != best_info.get("feature_names"):
        raise ValueError("Feature names changed after model selection; rerun model comparison")
    for name, df in (("train", train), ("validation", validation), ("test", test)):
        if len(df) != int(best_info[f"{name if name != 'train' else 'training'}_samples"]):
            raise ValueError(f"{name} sample count changed after model selection")
        if split_fingerprint(df) != best_info["split_fingerprints"][name]:
            raise ValueError(f"{name} IDs or labels changed after model selection")

    saved_model_path = model_dir / f"{best_name}.joblib"
    if not saved_model_path.is_file():
        raise FileNotFoundError(f"Selected model pipeline not found: {saved_model_path}")
    # clone preserves exactly the winner's hyperparameters and preprocessing
    # but resets learned parameters, so we can fit from scratch on TRAIN+VAL.
    selected_pipeline = clone(joblib.load(saved_model_path))
    X_train_val = pd.concat([train[features], validation[features]], ignore_index=True)
    y_train_val = pd.concat(
        [convert_labels(train["label"]), convert_labels(validation["label"])],
        ignore_index=True,
    )
    X_test = test[features]
    y_test = convert_labels(test["label"])
    print(f"Selected by validation F1: {best_name} "
          f"(F1={best_info['validation_f1_score']:.4f})")
    print(f"Fitting {best_name} on Train+Validation: {len(X_train_val)} videos")
    print(f"Test remains held out: {len(test)} videos")
    selected_pipeline.fit(X_train_val, y_train_val)
    # Not under models/: that folder is train_models' DVC output, and writing
    # into it here would make DVC see train_models as changed on every run.
    final_dir = Path("models_final")
    final_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_dir / f"final_{best_name}.joblib"
    joblib.dump(selected_pipeline, final_model_path)
    # The final .joblib file includes the imputer/scaler/classifier pipeline.
    # No separate scaler joblib is necessary for LR/SVM; RF has no scaler.

    predictions = selected_pipeline.predict(X_test)
    probabilities = None
    if hasattr(selected_pipeline, "predict_proba"):
        classes = list(selected_pipeline.classes_)
        probabilities = selected_pipeline.predict_proba(X_test)[:, classes.index(1)]
    roc_auc = (float(roc_auc_score(y_test, probabilities))
               if probabilities is not None and y_test.nunique() == 2 else None)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(y_test, predictions, labels=[0, 1],
                                   target_names=["real", "fake"], zero_division=0)
    metrics = {
        "model": best_name,
        "dataset": "full",
        "selection_metric": "validation_f1_score",
        "validation_f1_score_at_selection": float(best_info["validation_f1_score"]),
        "selected_features": len(features),
        "training_samples_original": len(train),
        "validation_samples": len(validation),
        "final_training_samples": len(X_train_val),
        "test_samples": len(test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }
    pred_df = pd.DataFrame({
        "media_id": test["media_id"].to_numpy(),
        "true_label": test["label"].to_numpy(),
        "true_label_binary": y_test.to_numpy(),
        "predicted_label_binary": predictions,
        "predicted_label": np.where(predictions == 1, "fake", "real"),
    })
    if probabilities is not None:
        pred_df["fake_probability"] = probabilities
        pred_df["real_probability"] = 1.0 - probabilities
    pred_df.to_csv(result_dir / "final_test_predictions.csv", index=False)
    (result_dir / "final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(cm, index=["Actual_Real", "Actual_Fake"],
                 columns=["Predicted_Real", "Predicted_Fake"]).to_csv(
                     result_dir / "final_test_confusion_matrix.csv"
                 )
    (result_dir / "final_test_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    final_report = (
        "FINAL MODEL EVALUATION - FULL DATASET\n"
        + "=" * 65 + "\n"
        + f"Selected by validation F1-score: {best_name}\n"
        + f"Validation F1 at selection: {best_info['validation_f1_score']:.4f}\n"
        + f"Original train: {len(train)} videos\n"
        + f"Validation: {len(validation)} videos\n"
        + f"Final model fit on Train+Validation: {len(X_train_val)} videos\n"
        + f"Held-out test: {len(test)} videos\n"
        + f"Selected feature count: {len(features)}\n"
        + "Model preprocessing and estimator refitted on Train+Validation only.\n"
        + "Previously selected feature list remains fixed.\n"
        + "Test was not used for model selection or fitting.\n\n"
        + "HELD-OUT TEST RESULTS\n"
        + f"Accuracy: {metrics['accuracy']:.4f}\n"
        + f"Precision: {metrics['precision']:.4f}\n"
        + f"Recall: {metrics['recall']:.4f}\n"
        + f"F1-score: {metrics['f1_score']:.4f}\n"
        + f"ROC-AUC: {roc_auc}\n\n"
        + f"Confusion matrix:\n{cm}\n\n{report}\n"
    )
    (result_dir / "final_evaluation_report.txt").write_text(
        final_report, encoding="utf-8"
    )
    print("\n" + final_report)
    print(f"Final fitted pipeline: {final_model_path}")
    print(f"Final test outputs: {result_dir}")


if __name__ == "__main__":
    main()
