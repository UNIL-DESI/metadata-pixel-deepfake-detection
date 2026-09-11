from __future__ import annotations

import argparse
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_dataset(path, name):
    print(f"\nLoading {name}...")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    df = pd.read_csv(path)
    print(f"{name} shape: {df.shape}")
    return df


def identify_columns(df):
    id_candidates = ["media_id", "video_name", "id"]
    label_candidates = ["label", "target", "class"]

    id_column = None
    label_column = None

    for col in id_candidates:
        if col in df.columns:
            id_column = col
            break

    for col in label_candidates:
        if col in df.columns:
            label_column = col
            break

    if id_column is None:
        raise ValueError("Could not find an identifier column.")

    if label_column is None:
        raise ValueError("Could not find a label column.")

    feature_columns = [
        col for col in df.columns
        if col not in [id_column, label_column]
    ]

    return id_column, label_column, feature_columns


def convert_labels(series):
    mapping = {
        "real": 0,
        "fake": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1
    }

    converted = series.map(mapping)

    if converted.isna().any():
        unexpected = set(
            series[converted.isna()].astype(str)
        )
        raise ValueError(
            f"Unexpected labels found: {unexpected}"
        )

    return converted.astype(int)


def check_dataset(df, name, id_column, label_column):
    print(f"\nChecking {name}...")

    if df[id_column].isna().any():
        raise ValueError(f"{name} contains missing IDs.")

    if df[id_column].duplicated().any():
        duplicates = df[
            df[id_column].duplicated(keep=False)
        ][id_column].tolist()
        raise ValueError(
            f"{name} contains duplicate IDs: {duplicates[:10]}"
        )

    if df[label_column].isna().any():
        raise ValueError(f"{name} contains missing labels.")

    labels = convert_labels(df[label_column])

    print("Label distribution:")
    print(
        pd.Series(labels)
        .map({0: "real", 1: "fake"})
        .value_counts()
    )

    print(f"✓ {name} passed validation checks.")


def check_split_overlap(train, validation, test, id_column):
    print("\nChecking ID overlap between splits...")

    train_ids = set(train[id_column])
    validation_ids = set(validation[id_column])
    test_ids = set(test[id_column])

    train_val = train_ids.intersection(validation_ids)
    train_test = train_ids.intersection(test_ids)
    val_test = validation_ids.intersection(test_ids)

    if train_val:
        raise ValueError(
            f"Train/Validation overlap detected: {list(train_val)[:10]}"
        )

    if train_test:
        raise ValueError(
            f"Train/Test overlap detected: {list(train_test)[:10]}"
        )

    if val_test:
        raise ValueError(
            f"Validation/Test overlap detected: {list(val_test)[:10]}"
        )

    print("✓ No ID overlap between train, validation and test.")


# ======================================================================
# MODEL DEFINITIONS
# ======================================================================

def create_models(random_state):
    models = {}

    models["logistic_regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=random_state
        ))
    ])

    models["random_forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    models["svm"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight="balanced"
        ))
    ])

    return models


# ======================================================================
# MODEL EVALUATION
# ======================================================================

def evaluate_model(
    model,
    X,
    y,
    model_name,
    split_name
):
    predictions = model.predict(X)
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    if probabilities is not None:
        auc = roc_auc_score(y, probabilities)
    else:
        auc = np.nan

    cm = confusion_matrix(y, predictions)

    print()
    print("-" * 60)
    print(f"{model_name.upper()} - {split_name.upper()}")
    print("-" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["real", "fake"],
            zero_division=0
        )
    )

    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1])
    }

    return metrics, predictions, probabilities


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Leakage-safe model training and validation")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config["dataset"]["base_dir"]
    selected_dir = str(Path(base_dir) / "features" / "selected")
    model_dir = str(Path(base_dir) / "models" / "pilot")
    result_dir = str(Path(base_dir) / "results" / "pilot")

    train_file = os.path.join(selected_dir, "train_selected_160.csv")
    validation_file = os.path.join(selected_dir, "validation_selected_160.csv")
    test_file = os.path.join(selected_dir, "test_selected_160.csv")

    random_state = 42

    print_header("MODEL TRAINING AND VALIDATION")
    print("Deepfake Detection - Pilot Dataset")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print_header("LOADING SELECTED FEATURES")

    train = load_dataset(train_file, "Training")
    validation = load_dataset(validation_file, "Validation")
    test = load_dataset(test_file, "Test")

    id_column, label_column, train_features = identify_columns(train)
    _, _, validation_features = identify_columns(validation)
    _, _, test_features = identify_columns(test)

    print("\nIdentifier column:")
    print(f"  {id_column}")

    print("\nLabel column:")
    print(f"  {label_column}")

    print(f"\nTraining feature count: {len(train_features)}")
    print(f"Validation feature count: {len(validation_features)}")
    print(f"Test feature count: {len(test_features)}")

    print_header("CHECKING FEATURE CONSISTENCY")

    if train_features != validation_features:
        raise ValueError(
            "Training and validation feature columns are not identical or in the same order."
        )

    if train_features != test_features:
        raise ValueError(
            "Training and test feature columns are not identical or in the same order."
        )

    print(f"✓ All splits contain the same {len(train_features)} features.")

    print_header("DATA VALIDATION")

    check_dataset(train, "Training", id_column, label_column)
    check_dataset(validation, "Validation", id_column, label_column)
    check_dataset(test, "Test", id_column, label_column)
    check_split_overlap(train, validation, test, id_column)

    X_train = train[train_features]
    X_validation = validation[train_features]
    X_test = test[train_features]

    y_train = convert_labels(train[label_column])
    y_validation = convert_labels(validation[label_column])
    y_test = convert_labels(test[label_column])

    models = create_models(random_state)

    all_results = []
    all_predictions = []

    print_header("TRAINING MODELS")

    for model_name, model in models.items():
        print()
        print("=" * 70)
        print(f"TRAINING: {model_name.upper()}")
        print("=" * 70)

        model.fit(X_train, y_train)
        print(f"✓ {model_name} training completed.")

        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"✓ Model saved:\n  {model_path}")

        metrics, predictions, probabilities = evaluate_model(
            model,
            X_validation,
            y_validation,
            model_name,
            "validation"
        )

        all_results.append(metrics)

        prediction_df = pd.DataFrame({
            "media_id": validation[id_column],
            "actual_label": validation[label_column],
            "predicted_label": [
                "fake" if p == 1 else "real"
                for p in predictions
            ]
        })

        if probabilities is not None:
            prediction_df["fake_probability"] = probabilities

        prediction_df["correct"] = (
            prediction_df["actual_label"] == prediction_df["predicted_label"]
        )

        prediction_path = os.path.join(
            result_dir,
            f"{model_name}_validation_predictions.csv"
        )
        prediction_df.to_csv(prediction_path, index=False)
        print(f"\n✓ Validation predictions saved:\n  {prediction_path}")

        all_predictions.append(prediction_df)

    print_header("MODEL COMPARISON")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    print(results_df[display_columns].to_string(index=False))

    comparison_path = os.path.join(result_dir, "validation_model_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison saved:\n  {comparison_path}")

    print_header("BEST MODEL BASED ON VALIDATION")

    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]

    print(f"Best model: {best_model_name}")
    print(f"Validation accuracy: {best_row['accuracy']:.4f}")
    print(f"Validation F1-score: {best_row['f1_score']:.4f}")
    print(f"Validation ROC-AUC: {best_row['roc_auc']:.4f}")

    best_model_info = {
        "best_model": best_model_name,
        "selection_metric": "f1_score",
        "validation_accuracy": float(best_row["accuracy"]),
        "validation_precision": float(best_row["precision"]),
        "validation_recall": float(best_row["recall"]),
        "validation_f1_score": float(best_row["f1_score"]),
        "validation_roc_auc": float(best_row["roc_auc"]),
        "training_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "feature_count": len(train_features),
        "random_state": random_state,
        "test_used": False
    }

    best_info_path = os.path.join(result_dir, "best_model_validation.json")
    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(best_model_info, f, indent=4)

    print(f"\n✓ Best model information saved:\n  {best_info_path}")

    summary_path = os.path.join(result_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DEEPFAKE DETECTION - PILOT MODEL TRAINING\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training samples: {len(train)}\n")
        f.write(f"Validation samples: {len(validation)}\n")
        f.write(f"Test samples: {len(test)}\n")
        f.write(f"Selected features: {len(train_features)}\n\n")
        f.write("Models trained:\n")
        for model_name in models:
            f.write(f"  - {model_name}\n")
        f.write("\nValidation results:\n")
        f.write(results_df[display_columns].to_string(index=False))
        f.write("\n\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write("Selection metric: F1-score\n")

    print(f"\n✓ Training summary saved:\n  {summary_path}")

    print_header("MODEL TRAINING COMPLETED")
    print(f"Training samples:       {len(train)}")
    print(f"Validation samples:     {len(validation)}")
    print(f"Test samples:           {len(test)} (NOT USED)")
    print(f"Selected features:      {len(train_features)}")
    print(f"\nBest validation model:  {best_model_name}")
    print(f"\nModels saved to:\n{model_dir}")
    print(f"\nResults saved to:\n{result_dir}")
    print("\nNext stage:\nFinal model evaluation on the TEST set.")


if __name__ == "__main__":
    main()