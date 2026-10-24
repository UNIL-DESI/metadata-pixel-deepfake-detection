from __future__ import annotations

"""Compare classifiers on TRAIN and VALIDATION; do not evaluate TEST."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Files are relative to the working directory (run from the project root).
DEFAULT_SELECTED_DIR = Path("features/selected")
DEFAULT_MODEL_DIR = Path("models/full")
DEFAULT_RESULT_DIR = Path("results/full")


def create_models(random_state: int = 42) -> dict[str, Pipeline]:
    """All preprocessing steps are part of each fitted model pipeline."""
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=500, random_state=random_state,
                n_jobs=-1, class_weight="balanced",
            )),
        ]),
        "svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", probability=True,
                random_state=random_state, class_weight="balanced",
            )),
        ]),
    }


def load_configuration(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file() and (Path("config") / path).is_file():
        config_path = Path("config") / path
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_directories(config: dict) -> tuple[Path, Path, Path]:
    # Optional config overrides; defaults target the requested 'full' experiment.
    paths = config.get("training", {})
    return (
        Path(paths.get("selected_dir", DEFAULT_SELECTED_DIR)),
        Path(paths.get("model_dir", DEFAULT_MODEL_DIR)),
        Path(paths.get("result_dir", DEFAULT_RESULT_DIR)),
    )


def convert_labels(labels: pd.Series) -> pd.Series:
    numeric = labels.astype(str).str.strip().str.lower().map(
        {"real": 0, "fake": 1, "0": 0, "1": 1}
    )
    if numeric.isna().any():
        raise ValueError(f"Unknown labels: {labels[numeric.isna()].unique()[:10].tolist()}")
    return numeric.astype(int)


def load_split(path: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} features: {path}")
    df = pd.read_csv(path, dtype={"media_id": str})
    if not {"media_id", "label"}.issubset(df.columns):
        raise ValueError(f"{path} needs media_id and label columns")
    if df["media_id"].isna().any() or df["media_id"].duplicated().any():
        raise ValueError(f"Missing/duplicate video IDs in {path}")
    if df["label"].isna().any():
        raise ValueError(f"Missing labels in {path}")
    convert_labels(df["label"])
    features = [c for c in df.columns if c not in {"media_id", "label"}]
    if not features:
        raise ValueError(f"No selected features in {path}")
    df[features] = df[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(df[features].to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite features in {path}")
    print(f"Loaded {split}: {len(df)} videos, {len(features)} features")
    return df, features


def validate_splits(train: pd.DataFrame, validation: pd.DataFrame,
                    test: pd.DataFrame, feature_lists: list[list[str]]) -> None:
    if not all(names == feature_lists[0] for names in feature_lists[1:]):
        raise ValueError("Selected feature columns must match, in the same order, across splits")
    id_sets = [set(df["media_id"]) for df in (train, validation, test)]
    if any(id_sets[i] & id_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Video-ID overlap between train, validation and test")


def split_fingerprint(df: pd.DataFrame) -> str:
    """Guard against evaluating a selection result from different split files."""
    records = sorted(zip(df["media_id"].astype(str), convert_labels(df["label"]).astype(str)))
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series,
                   model_name: str, split_name: str):
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probs = model.predict_proba(X)[:, classes.index(1)]
    auc = float(roc_auc_score(y, probs)) if probs is not None and y.nunique() == 2 else None
    cm = confusion_matrix(y, preds, labels=[0, 1])
    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1_score": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": auc,
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }
    report = classification_report(y, preds, labels=[0, 1],
                                   target_names=["real", "fake"], zero_division=0)
    return metrics, preds, probs, cm, report


def save_evaluation(result_dir: Path, model_name: str, split_name: str,
                    df: pd.DataFrame, metrics: dict, preds: np.ndarray,
                    probs: np.ndarray | None, cm: np.ndarray, report: str) -> None:
    stem = f"{model_name}_{split_name}"
    predictions_df = pd.DataFrame({
        "media_id": df["media_id"].to_numpy(),
        "actual_label": df["label"].to_numpy(),
        "predicted_label": np.where(preds == 1, "fake", "real"),
        "correct": convert_labels(df["label"]).to_numpy() == preds,
    })
    if probs is not None:
        predictions_df["fake_probability"] = probs
    predictions_df.to_csv(result_dir / f"{stem}_predictions.csv", index=False)
    pd.DataFrame(cm, index=["actual_real", "actual_fake"],
                 columns=["predicted_real", "predicted_fake"]).to_csv(
                     result_dir / f"{stem}_confusion_matrix.csv"
                 )
    (result_dir / f"{stem}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    (result_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/validation model comparison")
    parser.add_argument("--config", default="full.yaml")
    args = parser.parse_args()
    config = load_configuration(args.config)
    selected_dir, model_dir, result_dir = resolve_directories(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train, features = load_split(selected_dir / "train_selected_160.csv", "train")
    validation, validation_features = load_split(
        selected_dir / "validation_selected_160.csv", "validation")
    test, test_features = load_split(selected_dir / "test_selected_160.csv", "test")
    validate_splits(train, validation, test,
                    [features, validation_features, test_features])

    y_train, y_val = convert_labels(train["label"]), convert_labels(validation["label"])
    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both real and fake labels")
    X_train, X_val = train[features], validation[features]
    random_state = int(config.get("training", {}).get("random_state", 42))
    models = create_models(random_state)
    all_metrics = []

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        joblib.dump(model, model_dir / f"{name}.joblib")
        for split_name, df, X, y in (
            ("train", train, X_train, y_train),
            ("validation", validation, X_val, y_val),
        ):
            metrics, predictions, probs, cm, report = evaluate_model(
                model, X, y, name, split_name
            )
            all_metrics.append(metrics)
            save_evaluation(result_dir, name, split_name, df, metrics,
                            predictions, probs, cm, report)
            print(f"  {split_name}: F1={metrics['f1_score']:.4f}, "
                  f"accuracy={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']}")

    metrics_df = pd.DataFrame(all_metrics)
    validation_results = (
        metrics_df[metrics_df["split"] == "validation"]
        .sort_values(by=["f1_score", "model"], ascending=[False, True])
        .reset_index(drop=True)
    )
    train_results = (
        metrics_df[metrics_df["split"] == "train"]
        .set_index("model").loc[validation_results["model"]]
        .reset_index()
    )
    metrics_df.to_csv(result_dir / "model_metrics_summary.csv", index=False)
    validation_results.to_csv(result_dir / "validation_model_comparison.csv", index=False)
    train_results.to_csv(result_dir / "training_model_comparison.csv", index=False)

    winner = validation_results.iloc[0]
    best_info = {
        "best_model": str(winner["model"]),
        "selection_metric": "validation_f1_score",
        "tie_breaker": "model_name_ascending",
        "validation_accuracy": float(winner["accuracy"]),
        "validation_precision": float(winner["precision"]),
        "validation_recall": float(winner["recall"]),
        "validation_f1_score": float(winner["f1_score"]),
        "validation_roc_auc": None if pd.isna(winner["roc_auc"]) else float(winner["roc_auc"]),
        "training_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "feature_count": len(features),
        "feature_names": features,
        "split_fingerprints": {
            "train": split_fingerprint(train),
            "validation": split_fingerprint(validation),
            "test": split_fingerprint(test),
        },
        "random_state": random_state,
        "test_used": False,
    }
    (result_dir / "best_model_validation.json").write_text(
        json.dumps(best_info, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    summary = (
        "FULL DATASET - MODEL TRAINING AND VALIDATION\n\n"
        f"Training videos: {len(train)}\nValidation videos: {len(validation)}\n"
        f"Test videos: {len(test)} (not evaluated)\n"
        f"Selected features: {len(features)}\n\n"
        "Training metrics (in-sample, diagnostic only):\n"
        + train_results[columns].to_string(index=False)
        + "\n\nValidation metrics (used for model selection):\n"
        + validation_results[columns].to_string(index=False)
        + f"\n\nBest model by validation F1-score: {winner['model']}\n"
        + f"Best validation F1: {winner['f1_score']:.4f}\n"
        + "TEST was not used for fitting, ranking, or model selection.\n"
    )
    (result_dir / "training_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved model pipelines to {model_dir}")
    print(f"Saved training/validation metrics to {result_dir}")


if __name__ == "__main__":
    main()
