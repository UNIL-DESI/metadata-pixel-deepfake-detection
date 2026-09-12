from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import yaml

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

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(path, name):
    print(f"\nLoading {name}...")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)
    print(f"{name} shape: {df.shape}")
    return df


def convert_labels(series):
    mapping = {
        "real": 0,
        "fake": 1,
        0: 0,
        1: 1
    }

    converted = series.map(mapping)

    if converted.isna().any():
        unknown = series[converted.isna()].unique()
        raise ValueError(
            f"Unknown labels found: {unknown}"
        )

    return converted.astype(int)


def validate_dataset(df, name):
    print(f"\nChecking {name}...")

    if "media_id" not in df.columns:
        raise ValueError(
            f"'media_id' column missing from {name}"
        )

    if "label" not in df.columns:
        raise ValueError(
            f"'label' column missing from {name}"
        )

    print("Label distribution:")
    print(df["label"].value_counts())

    if df["media_id"].duplicated().any():
        duplicates = df.loc[
            df["media_id"].duplicated(),
            "media_id"
        ].tolist()

        raise ValueError(
            f"Duplicate media IDs found in {name}: {duplicates[:10]}"
        )

    print(f"✓ {name} passed validation checks.")


def check_id_overlap(train_df, validation_df, test_df):
    train_ids = set(train_df["media_id"])
    validation_ids = set(validation_df["media_id"])
    test_ids = set(test_df["media_id"])

    train_validation_overlap = (
        train_ids.intersection(validation_ids)
    )

    train_test_overlap = (
        train_ids.intersection(test_ids)
    )

    validation_test_overlap = (
        validation_ids.intersection(test_ids)
    )

    if train_validation_overlap:
        raise ValueError(
            "ID overlap detected between TRAIN and VALIDATION."
        )

    if train_test_overlap:
        raise ValueError(
            "ID overlap detected between TRAIN and TEST."
        )

    if validation_test_overlap:
        raise ValueError(
            "ID overlap detected between VALIDATION and TEST."
        )

    print("✓ No ID overlap between train, validation and test.")


def load_selected_feature_names(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Selected feature list not found:\n{path}"
        )

    features = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ". " in line:
                prefix, remainder = line.split(
                    ". ",
                    1
                )

                if prefix.isdigit():
                    line = remainder

            features.append(line)

    return features


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Final model evaluation")
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
    selected_dir = "features/selected"
    model_output_dir = "models/pilot"
    results_output_dir = "results/pilot"

    train_file = os.path.join(selected_dir, "train_selected_160.csv")
    validation_file = os.path.join(selected_dir, "validation_selected_160.csv")
    test_file = os.path.join(selected_dir, "test_selected_160.csv")
    selected_features_file = os.path.join(selected_dir, "selected_features_160.txt")

    final_model_file = os.path.join(model_output_dir, "final_logistic_regression.joblib")
    scaler_file = os.path.join(model_output_dir, "final_logistic_regression_scaler.joblib")
    test_predictions_file = os.path.join(results_output_dir, "final_test_predictions.csv")
    test_metrics_file = os.path.join(results_output_dir, "final_test_metrics.json")
    confusion_matrix_file = os.path.join(results_output_dir, "final_test_confusion_matrix.csv")
    classification_report_file = os.path.join(results_output_dir, "final_test_classification_report.txt")
    final_report_file = os.path.join(results_output_dir, "final_evaluation_report.txt")

    print_section("FINAL MODEL EVALUATION")
    print("Deepfake Detection - Pilot Dataset")
    print("\nSelected model:\n  Logistic Regression")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    print_section("LOADING SELECTED FEATURES")

    train_df = load_csv(train_file, "Training")
    validation_df = load_csv(validation_file, "Validation")
    test_df = load_csv(test_file, "Test")

    print_section("DATA VALIDATION")
    validate_dataset(train_df, "Training")
    validate_dataset(validation_df, "Validation")
    validate_dataset(test_df, "Test")

    print_section("CHECKING DATASET ID SEPARATION")
    check_id_overlap(train_df, validation_df, test_df)

    print_section("LOADING FIXED 160 FEATURES")
    selected_features = load_selected_feature_names(selected_features_file)
    print(f"Features listed in file: {len(selected_features)}")

    if len(selected_features) != 160:
        raise ValueError(
            f"Expected exactly 160 selected features, but found {len(selected_features)}."
        )

    print("✓ Exactly 160 selected features loaded.")

    print_section("CHECKING FEATURE CONSISTENCY")
    for feature in selected_features:
        if feature not in train_df.columns:
            raise ValueError(f"Feature '{feature}' missing from training data.")
        if feature not in validation_df.columns:
            raise ValueError(f"Feature '{feature}' missing from validation data.")
        if feature not in test_df.columns:
            raise ValueError(f"Feature '{feature}' missing from test data.")

    print("✓ All 160 features are present in all splits.")

    print_section("PREPARING LABELS")
    y_train = convert_labels(train_df["label"])
    y_validation = convert_labels(validation_df["label"])
    y_test = convert_labels(test_df["label"])

    print("\nTraining labels:")
    print(pd.Series(y_train).value_counts())
    print("\nValidation labels:")
    print(pd.Series(y_validation).value_counts())
    print("\nTest labels:")
    print(pd.Series(y_test).value_counts())

    print_section("PREPARING FEATURE MATRICES")
    X_train = train_df[selected_features].copy()
    X_validation = validation_df[selected_features].copy()
    X_test = test_df[selected_features].copy()

    print(f"X_train:       {X_train.shape}")
    print(f"X_validation:  {X_validation.shape}")
    print(f"X_test:        {X_test.shape}")

    print_section("CHECKING FEATURE VALUES")
    for name, X in [
        ("Training", X_train),
        ("Validation", X_validation),
        ("Test", X_test)
    ]:
        missing = X.isna().sum().sum()
        infinite = np.isinf(
            X.select_dtypes(
                include=[np.number]
            ).to_numpy()
        ).sum()

        print(f"{name}:")
        print(f"  Missing values:   {missing}")
        print(f"  Infinite values:  {infinite}")

        if missing > 0:
            raise ValueError(f"Missing values detected in {name}.")
        if infinite > 0:
            raise ValueError(f"Infinite values detected in {name}.")

    print("\n✓ Feature values passed validation.")

    print_section("CREATING FINAL TRAINING DATA")
    X_train_final = pd.concat(
        [X_train, X_validation],
        axis=0,
        ignore_index=True
    )

    y_train_final = pd.concat(
        [pd.Series(y_train), pd.Series(y_validation)],
        axis=0,
        ignore_index=True
    )

    print(f"Original training samples:     {len(X_train)}")
    print(f"Validation samples:            {len(X_validation)}")
    print(f"Final training samples:        {len(X_train_final)}")
    print(f"Final test samples:            {len(X_test)}")
    print("\nFinal training label distribution:")
    print(y_train_final.value_counts())

    print_section("FITTING FEATURE STANDARDIZATION")
    print("StandardScaler is fitted ONLY on:\n  Training + Validation data")
    print("Test data is NOT used to fit the scaler.")

    scaler = StandardScaler()
    X_train_final_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test)

    print("\n✓ Standardization completed.")

    print_section("TRAINING FINAL LOGISTIC REGRESSION")
    print(f"Training samples: {X_train_final_scaled.shape[0]}")
    print(f"Features: {X_train_final_scaled.shape[1]}")
    print("\nModel:\n  Logistic Regression\n  max_iter = 2000")

    final_model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    final_model.fit(
        X_train_final_scaled,
        y_train_final
    )

    print("\n✓ Final Logistic Regression training completed.")

    print_section("SAVING FINAL MODEL")
    joblib.dump(final_model, final_model_file)
    joblib.dump(scaler, scaler_file)

    print(f"Model saved:\n{final_model_file}")
    print(f"\nScaler saved:\n{scaler_file}")

    print_section("FINAL TEST EVALUATION")
    print("IMPORTANT:\nThis is the FIRST time the test set is being used.")
    print("The test set was NOT used for:\n  - feature selection\n  - preprocessing fitting\n  - model training\n  - model selection")

    y_test_pred = final_model.predict(X_test_scaled)
    y_test_probability = final_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_probability)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(
        y_test,
        y_test_pred,
        target_names=["real", "fake"],
        zero_division=0
    )

    print_section("FINAL TEST RESULTS")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    print_section("SAVING TEST PREDICTIONS")
    prediction_df = pd.DataFrame({
        "media_id": test_df["media_id"].values,
        "true_label": test_df["label"].values,
        "true_label_binary": y_test,
        "predicted_label_binary": y_test_pred,
        "predicted_label": [
            "fake" if p == 1 else "real"
            for p in y_test_pred
        ],
        "fake_probability": y_test_probability,
        "real_probability": (1 - y_test_probability)
    })

    prediction_df.to_csv(test_predictions_file, index=False)
    print(f"Test predictions saved:\n{test_predictions_file}")

    metrics = {
        "model": "logistic_regression",
        "dataset": "pilot",
        "selected_features": 160,
        "training_samples_original": int(len(X_train)),
        "validation_samples": int(len(X_validation)),
        "final_training_samples": int(len(X_train_final)),
        "test_samples": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }

    with open(test_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTest metrics saved:\n{test_metrics_file}")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Real", "Actual_Fake"],
        columns=["Predicted_Real", "Predicted_Fake"]
    )
    cm_df.to_csv(confusion_matrix_file)
    print(f"\nConfusion matrix saved:\n{confusion_matrix_file}")

    with open(classification_report_file, "w", encoding="utf-8") as f:
        f.write("FINAL TEST CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    print(f"\nClassification report saved:\n{classification_report_file}")

    print_section("CREATING FINAL EVALUATION REPORT")
    report_lines = [
        "FINAL MODEL EVALUATION REPORT",
        "=" * 70,
        "",
        "Deepfake Detection - Pilot Dataset",
        "",
        "MODEL",
        "-" * 70,
        "Selected model: Logistic Regression",
        "",
        "FEATURE SELECTION",
        "-" * 70,
        "Final feature count: 160",
        "Feature selection performed using training data only.",
        "The selected 160 features were kept fixed for final evaluation.",
        "",
        "DATA SPLIT",
        "-" * 70,
        "Original training samples: 210",
        "Validation samples: 45",
        "Test samples: 45",
        "Final model training samples: 255",
        "",
        "FINAL TRAINING",
        "-" * 70,
        "The final Logistic Regression model was trained using",
        "the original training + validation datasets.",
        "",
        "PREPROCESSING",
        "-" * 70,
        "StandardScaler was fitted using training + validation data only.",
        "The test set was transformed using the fitted scaler.",
        "",
        "TEST SET",
        "-" * 70,
        "The test set was kept completely untouched until final evaluation.",
        "",
        "FINAL TEST PERFORMANCE",
        "-" * 70,
        f"Accuracy:   {accuracy:.4f}",
        f"Precision:  {precision:.4f}",
        f"Recall:     {recall:.4f}",
        f"F1-score:   {f1:.4f}",
        f"ROC-AUC:    {roc_auc:.4f}",
        "",
        "CONFUSION MATRIX",
        "-" * 70,
        str(cm),
        "",
        "CLASSIFICATION REPORT",
        "-" * 70,
        report,
        "",
        "METHODOLOGICAL NOTE",
        "-" * 70,
        "The test set was not used for feature selection,",
        "preprocessing fitting, model training, or model selection.",
        "Therefore, the reported test performance represents",
        "the final evaluation of the selected model on unseen data.",
        ""
    ]

    with open(final_report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Final evaluation report saved:\n{final_report_file}")

    print_section("FINAL EVALUATION COMPLETED")
    print(f"Final training samples: {len(X_train_final)}")
    print(f"Test samples:           {len(X_test)}")
    print(f"Selected features:      {len(selected_features)}")
    print("\nFinal model:\nLogistic Regression")
    print(f"\nFinal Test Accuracy:\n{accuracy:.4f}")
    print(f"\nFinal Test F1-score:\n{f1:.4f}")
    print(f"\nFinal Test ROC-AUC:\n{roc_auc:.4f}")
    print(f"\nModel saved to:\n{model_output_dir}")
    print(f"\nResults saved to:\n{results_output_dir}")
    print("\n✓ Test set was used only for final evaluation.")
    print("✓ Final evaluation completed.")


if __name__ == "__main__":
    main()