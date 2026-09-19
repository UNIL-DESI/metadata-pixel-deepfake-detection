from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore")


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def identify_feature_group(column):
    if column.startswith("hc_"):
        return "hc"
    if column.startswith("cnn_"):
        return "cnn"
    if column.startswith("meta_"):
        return "meta"
    return "other"


def get_group_features(columns, groups):
    selected = []
    for col in columns:
        group = identify_feature_group(col)
        if group in groups:
            selected.append(col)
    return selected


def remove_constant_features(X_train, feature_columns):
    variances = X_train[feature_columns].var(axis=0)
    constant_features = variances[variances == 0].index.tolist()
    remaining_features = [
        col for col in feature_columns
        if col not in constant_features
    ]
    return remaining_features, constant_features


def remove_highly_correlated_features(
    X_train,
    feature_columns,
    threshold=0.95
):
    if len(feature_columns) <= 1:
        return feature_columns, []

    corr_matrix = X_train[feature_columns].corr().abs()

    upper = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    to_remove = [
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    ]

    remaining = [
        column
        for column in feature_columns
        if column not in to_remove
    ]

    return remaining, to_remove


def calculate_mutual_information(
    X_train,
    y_train,
    feature_columns,
    random_state
):
    if len(feature_columns) == 0:
        return pd.DataFrame(
            columns=["feature", "mutual_information"]
        )

    X = X_train[feature_columns]

    mi_values = mutual_info_classif(
        X,
        y_train,
        random_state=random_state
    )

    mi_df = pd.DataFrame({
        "feature": feature_columns,
        "mutual_information": mi_values
    })

    mi_df = mi_df.sort_values(
        "mutual_information",
        ascending=False
    ).reset_index(drop=True)

    return mi_df


def select_features_by_mi(
    mi_df,
    target_features
):
    n = min(target_features, len(mi_df))
    return mi_df.head(n)["feature"].tolist()


def save_feature_list(
    output_dir,
    experiment_name,
    selected_features,
    all_candidates,
    constant_removed,
    correlated_removed,
):
    experiment_dir = os.path.join(
        output_dir,
        experiment_name
    )

    os.makedirs(experiment_dir, exist_ok=True)

    with open(
        os.path.join(
            experiment_dir,
            "selected_features.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        for feature in selected_features:
            f.write(feature + "\n")

    with open(
        os.path.join(
            experiment_dir,
            "feature_selection_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write("FEATURE SELECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Feature selection was performed using TRAINING data only.\n\n")
        f.write(f"Initial candidate features: {len(all_candidates)}\n")
        f.write(f"Constant features removed: {len(constant_removed)}\n")
        f.write(f"Highly correlated features removed: {len(correlated_removed)}\n")
        f.write(
            f"Features after cleaning: "
            f"{len(all_candidates) - len(constant_removed) - len(correlated_removed)}\n"
        )
        f.write(f"Final selected features: {len(selected_features)}\n\n")


def calculate_metrics(y_true, y_pred, y_probability):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability
        ),
    }


def train_validation_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    random_state
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = LogisticRegression(
        max_iter=2000,
        random_state=random_state
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_validation_scaled
    )

    y_probability = model.predict_proba(
        X_validation_scaled
    )[:, 1]

    metrics = calculate_metrics(
        y_validation,
        y_pred,
        y_probability
    )

    return model, scaler, metrics


def final_train_and_test(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    random_state
):
    X_final_train = pd.concat(
        [X_train, X_validation],
        axis=0
    )

    y_final_train = pd.concat(
        [y_train, y_validation],
        axis=0
    )

    scaler = StandardScaler()
    X_final_scaled = scaler.fit_transform(
        X_final_train
    )
    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=random_state
    )

    model.fit(
        X_final_scaled,
        y_final_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability
    )

    return (
        model,
        scaler,
        metrics,
        y_pred,
        y_probability
    )


def save_confusion_matrix(
    cm,
    output_dir,
    experiment_name,
    stage
):
    experiment_dir = os.path.join(
        output_dir,
        experiment_name
    )

    cm_df = pd.DataFrame(
        cm,
        index=["real", "fake"],
        columns=["real", "fake"]
    )

    cm_df.to_csv(
        os.path.join(
            experiment_dir,
            f"{stage}_confusion_matrix.csv"
        )
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(
        f"{experiment_name} - {stage.replace('_', ' ').title()}"
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            experiment_dir,
            f"{stage}_confusion_matrix.png"
        ),
        dpi=200
    )
    plt.close()


def save_classification_report(
    y_true,
    y_pred,
    output_dir,
    experiment_name,
    stage
):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["real", "fake"],
        zero_division=0
    )

    experiment_dir = os.path.join(
        output_dir,
        experiment_name
    )

    with open(
        os.path.join(
            experiment_dir,
            f"{stage}_classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe ablation study")
    parser.add_argument(
        "--config",
        default="full.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config["dataset"]["base_dir"]
    input_file = "features/aligned/aligned_all_features.csv"
    output_dir = "results/full/ablation"

    random_state = 42
    validation_size = 0.15
    test_size = 0.15
    target_features = 160
    correlation_threshold = 0.95

    experiments = {
        "E1_Metadata": ["meta"],
        "E2_Handcrafted": ["hc"],
        "E3_CNN": ["cnn"],
        "E4_Handcrafted_CNN": ["hc", "cnn"],
        "E5_All": ["hc", "cnn", "meta"],
    }

    print_header("ABLATION STUDY")
    print("Deepfake Detection - Pilot Dataset")

    os.makedirs(output_dir, exist_ok=True)

    print_header("LOADING ALIGNED DATA")
    print(f"Input file:\n{input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"\nInput file not found:\n{input_file}")

    df = pd.read_csv(input_file)
    print(f"\nDataset shape: {df.shape}")

    if "media_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Required columns missing.")

    train_df, temp_df = train_test_split(
        df,
        test_size=(validation_size + test_size),
        stratify=df["label"],
        random_state=random_state
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (validation_size + test_size),
        stratify=temp_df["label"],
        random_state=random_state
    )

    label_mapping = {"real": 0, "fake": 1}
    train_y = train_df["label"].map(label_mapping)
    validation_y = validation_df["label"].map(label_mapping)
    test_y = test_df["label"].map(label_mapping)

    all_feature_columns = [
        col for col in df.columns
        if col not in ["media_id", "label"]
    ]

    all_validation_results = []
    all_test_results = []
    roc_curves = []
    pr_curves = []

    for experiment_name, groups in experiments.items():
        print_header(f"{experiment_name.upper()}")

        candidate_features = get_group_features(all_feature_columns, groups)
        if len(candidate_features) == 0:
            continue

        X_train_raw = train_df[candidate_features].copy()
        X_validation_raw = validation_df[candidate_features].copy()
        X_test_raw = test_df[candidate_features].copy()

        cleaned_features, constant_removed = remove_constant_features(
            X_train_raw, candidate_features
        )
        cleaned_features, correlated_removed = remove_highly_correlated_features(
            X_train_raw, cleaned_features, correlation_threshold
        )

        mi_df = calculate_mutual_information(
            X_train_raw, train_y, cleaned_features, random_state
        )

        selected_features = select_features_by_mi(mi_df, target_features)

        save_feature_list(
            output_dir,
            experiment_name,
            selected_features,
            candidate_features,
            constant_removed,
            correlated_removed
        )

        X_train_selected = train_df[selected_features].copy()
        X_validation_selected = validation_df[selected_features].copy()
        X_test_selected = test_df[selected_features].copy()

        validation_model, validation_scaler, validation_metrics = train_validation_model(
            X_train_selected,
            train_y,
            X_validation_selected,
            validation_y,
            random_state
        )

        validation_predictions = validation_model.predict(
            validation_scaler.transform(X_validation_selected)
        )
        validation_probability = validation_model.predict_proba(
            validation_scaler.transform(X_validation_selected)
        )[:, 1]

        validation_cm = confusion_matrix(validation_y, validation_predictions)
        save_confusion_matrix(validation_cm, output_dir, experiment_name, "validation")
        save_classification_report(validation_y, validation_predictions, output_dir, experiment_name, "validation")

        validation_result = {
            "experiment": experiment_name,
            "feature_groups": "+".join(groups),
            "selected_features": len(selected_features),
            **validation_metrics
        }
        all_validation_results.append(validation_result)

        (
            final_model,
            final_scaler,
            test_metrics,
            test_predictions,
            test_probability
        ) = final_train_and_test(
            X_train_selected,
            train_y,
            X_validation_selected,
            validation_y,
            X_test_selected,
            test_y,
            random_state
        )

        test_cm = confusion_matrix(test_y, test_predictions)
        save_confusion_matrix(test_cm, output_dir, experiment_name, "test")
        save_classification_report(test_y, test_predictions, output_dir, experiment_name, "test")

        test_result = {
            "experiment": experiment_name,
            "feature_groups": "+".join(groups),
            "selected_features": len(selected_features),
            **test_metrics
        }
        all_test_results.append(test_result)

        fpr, tpr, _ = roc_curve(test_y, test_probability)
        roc_curves.append({
            "experiment": experiment_name,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr)
        })

        precision, recall, _ = precision_recall_curve(test_y, test_probability)
        pr_curves.append({
            "experiment": experiment_name,
            "precision": precision,
            "recall": recall,
            "auc": auc(recall, precision)
        })

    validation_results_df = pd.DataFrame(all_validation_results)
    test_results_df = pd.DataFrame(all_test_results)

    validation_results_df.to_csv(
        os.path.join(output_dir, "ablation_validation_results.csv"), index=False
    )
    test_results_df.to_csv(
        os.path.join(output_dir, "ablation_test_results.csv"), index=False
    )

    print_header("ABLATION STUDY COMPLETED")
    print(f"Output directory:\n{output_dir}")


if __name__ == "__main__":
    main()
    #