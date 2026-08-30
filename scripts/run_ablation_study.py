"""
run_ablation_study.py

Leakage-safe ablation study for the Deepfake Detection pilot dataset.

Experiments:
    E1 - Metadata only
    E2 - Handcrafted only
    E3 - CNN only
    E4 - Handcrafted + CNN
    E5 - Metadata + Handcrafted + CNN

Important:
    - Same fixed 70/15/15 split is used for every experiment.
    - Feature selection is performed separately inside each experiment.
    - Feature selection uses TRAINING data only.
    - StandardScaler is fitted on TRAINING data only during validation.
    - Validation is used for model comparison.
    - After comparison, the selected feature set is refitted using
      TRAIN + VALIDATION.
    - TEST is used only for final evaluation.
    - Logistic Regression is used for all experiments so that the
      ablation measures feature-group contribution rather than
      classifier differences.

Output:
    G:/My Drive/DeepfakeResearch/results/pilot/ablation/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    r"G:\My Drive\DeepfakeResearch"
    r"\features\aligned\aligned_all_features.csv"
)

OUTPUT_DIR = (
    r"G:\My Drive\DeepfakeResearch"
    r"\results\pilot\ablation"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

TARGET_FEATURES = 160

CORRELATION_THRESHOLD = 0.95

RF_ESTIMATORS = 300


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {
    "E1_Metadata": ["meta"],
    "E2_Handcrafted": ["hc"],
    "E3_CNN": ["cnn"],
    "E4_Handcrafted_CNN": ["hc", "cnn"],
    "E5_All": ["hc", "cnn", "meta"],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def identify_feature_group(column):
    """
    Determine feature group from the aligned feature naming convention.
    """

    if column.startswith("hc_"):
        return "hc"

    if column.startswith("cnn_"):
        return "cnn"

    if column.startswith("meta_"):
        return "meta"

    return "other"


def get_group_features(columns, groups):
    """
    Return feature columns belonging to selected feature groups.
    """

    selected = []

    for col in columns:
        group = identify_feature_group(col)

        if group in groups:
            selected.append(col)

    return selected


def remove_constant_features(X_train, feature_columns):
    """
    Remove features that have zero variance in TRAINING data.

    This is intentionally fitted using TRAINING data only.
    """

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
    """
    Remove highly correlated features using TRAINING data only.

    For every highly correlated pair, the later feature is removed.
    """

    if len(feature_columns) <= 1:
        return feature_columns, []

    print(
        f"Calculating correlation matrix for "
        f"{len(feature_columns)} features..."
    )

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
    feature_columns
):
    """
    Calculate mutual information using TRAINING data only.
    """

    if len(feature_columns) == 0:
        return pd.DataFrame(
            columns=["feature", "mutual_information"]
        )

    print(
        f"Calculating mutual information for "
        f"{len(feature_columns)} features..."
    )

    X = X_train[feature_columns]

    mi_values = mutual_info_classif(
        X,
        y_train,
        random_state=RANDOM_STATE
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
    """
    Select the top N features according to mutual information.

    If fewer than target_features are available,
    all available features are selected.
    """

    n = min(target_features, len(mi_df))

    return mi_df.head(n)["feature"].tolist()


def save_feature_list(
    experiment_name,
    selected_features,
    all_candidates,
    constant_removed,
    correlated_removed,
):
    """
    Save feature-selection information for reproducibility.
    """

    experiment_dir = os.path.join(
        OUTPUT_DIR,
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

        f.write(
            "Feature selection was performed using TRAINING data only.\n\n"
        )

        f.write(
            f"Initial candidate features: {len(all_candidates)}\n"
        )

        f.write(
            f"Constant features removed: {len(constant_removed)}\n"
        )

        f.write(
            f"Highly correlated features removed: "
            f"{len(correlated_removed)}\n"
        )

        f.write(
            f"Features after cleaning: "
            f"{len(all_candidates) - len(constant_removed) - len(correlated_removed)}\n"
        )

        f.write(
            f"Final selected features: {len(selected_features)}\n\n"
        )

        f.write("SELECTED FEATURES\n")
        f.write("-" * 70 + "\n")

        for feature in selected_features:
            f.write(feature + "\n")

        f.write("\nREMOVED CONSTANT FEATURES\n")
        f.write("-" * 70 + "\n")

        for feature in constant_removed:
            f.write(feature + "\n")

        f.write("\nREMOVED HIGHLY CORRELATED FEATURES\n")
        f.write("-" * 70 + "\n")

        for feature in correlated_removed:
            f.write(feature + "\n")


def calculate_metrics(y_true, y_pred, y_probability):
    """
    Calculate classification metrics.
    """

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
    y_validation
):
    """
    Fit scaler ONLY on training data,
    then train Logistic Regression.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(X_validation)

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
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
    y_test
):
    """
    Final training:

        TRAIN + VALIDATION -> fit scaler
        TRAIN + VALIDATION -> fit model
        TEST -> final evaluation
    """

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
        random_state=RANDOM_STATE
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
    experiment_name,
    stage
):
    """
    Save confusion matrix as CSV and PNG.
    """

    experiment_dir = os.path.join(
        OUTPUT_DIR,
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

    plt.xticks(
        [0, 1],
        ["Real", "Fake"]
    )

    plt.yticks(
        [0, 1],
        ["Real", "Fake"]
    )

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
        OUTPUT_DIR,
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


# ============================================================
# LOAD DATA
# ============================================================

print_header("ABLATION STUDY")
print("Deepfake Detection - Pilot Dataset")

print("\nExperiments:")

for name, groups in EXPERIMENTS.items():
    print(f"  {name}: {', '.join(groups)}")

print("\nIMPORTANT:")
print("  ✓ Same dataset split used for all experiments.")
print("  ✓ Feature selection is performed separately.")
print("  ✓ Feature selection uses TRAINING data only.")
print("  ✓ Validation is used for model comparison.")
print("  ✓ Test is used only for final evaluation.")
print("  ✓ Logistic Regression is used for every experiment.")

ensure_output_directory()

print_header("LOADING ALIGNED DATA")

print(f"Input file:\n{INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")

if "media_id" not in df.columns:
    raise ValueError(
        "media_id column was not found."
    )

if "label" not in df.columns:
    raise ValueError(
        "label column was not found."
    )

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# CHECK IDs
# ============================================================

if df["media_id"].duplicated().any():

    duplicated_ids = (
        df.loc[
            df["media_id"].duplicated(),
            "media_id"
        ]
        .tolist()
    )

    raise ValueError(
        f"Duplicate media IDs found: {duplicated_ids[:10]}"
    )


# ============================================================
# CREATE FIXED SPLIT
# ============================================================

print_header("CREATING FIXED 70 / 15 / 15 SPLIT")

train_df, temp_df = train_test_split(
    df,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    stratify=df["label"],
    random_state=RANDOM_STATE
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)

print(f"Train:       {len(train_df)}")
print(f"Validation:  {len(validation_df)}")
print(f"Test:        {len(test_df)}")

print("\nTrain labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(validation_df["label"].value_counts())

print("\nTest labels:")
print(test_df["label"].value_counts())


# ============================================================
# CHECK SPLIT ID OVERLAP
# ============================================================

train_ids = set(train_df["media_id"])
validation_ids = set(validation_df["media_id"])
test_ids = set(test_df["media_id"])

if train_ids & validation_ids:
    raise ValueError(
        "ID overlap detected between TRAIN and VALIDATION."
    )

if train_ids & test_ids:
    raise ValueError(
        "ID overlap detected between TRAIN and TEST."
    )

if validation_ids & test_ids:
    raise ValueError(
        "ID overlap detected between VALIDATION and TEST."
    )

print("\n✓ No ID overlap between splits.")


# ============================================================
# LABEL ENCODING
# ============================================================

label_mapping = {
    "real": 0,
    "fake": 1
}

train_y = train_df["label"].map(label_mapping)
validation_y = validation_df["label"].map(label_mapping)
test_y = test_df["label"].map(label_mapping)

if train_y.isna().any() or validation_y.isna().any() or test_y.isna().any():
    raise ValueError(
        "Unexpected label value detected."
    )


# ============================================================
# IDENTIFY FEATURE GROUPS
# ============================================================

all_feature_columns = [
    col
    for col in df.columns
    if col not in ["media_id", "label"]
]

group_counts = {}

for group in ["hc", "cnn", "meta", "other"]:

    group_counts[group] = sum(
        identify_feature_group(col) == group
        for col in all_feature_columns
    )

print_header("FEATURE GROUP SUMMARY")

print(f"Handcrafted features: {group_counts['hc']}")
print(f"CNN features:         {group_counts['cnn']}")
print(f"Metadata features:    {group_counts['meta']}")
print(f"Other features:       {group_counts['other']}")


# ============================================================
# RUN ABLATION EXPERIMENTS
# ============================================================

all_validation_results = []
all_test_results = []

roc_curves = []
pr_curves = []


for experiment_name, groups in EXPERIMENTS.items():

    print_header(
        f"{experiment_name.upper()}"
    )

    print(
        f"Feature groups: {', '.join(groups)}"
    )

    # --------------------------------------------------------
    # CREATE EXPERIMENT DIRECTORY
    # --------------------------------------------------------

    experiment_dir = os.path.join(
        OUTPUT_DIR,
        experiment_name
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SELECT CANDIDATE FEATURES
    # --------------------------------------------------------

    candidate_features = get_group_features(
        all_feature_columns,
        groups
    )

    print(
        f"\nCandidate features: "
        f"{len(candidate_features)}"
    )

    if len(candidate_features) == 0:
        print(
            "No features available for this experiment."
        )
        continue

    # --------------------------------------------------------
    # TRAINING DATA ONLY
    # --------------------------------------------------------

    X_train_raw = train_df[candidate_features].copy()

    X_validation_raw = validation_df[
        candidate_features
    ].copy()

    X_test_raw = test_df[
        candidate_features
    ].copy()

    # --------------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # --------------------------------------------------------

    print("\nRemoving constant features...")

    cleaned_features, constant_removed = (
        remove_constant_features(
            X_train_raw,
            candidate_features
        )
    )

    print(
        f"Constant features removed: "
        f"{len(constant_removed)}"
    )

    # --------------------------------------------------------
    # REMOVE HIGHLY CORRELATED FEATURES
    # --------------------------------------------------------

    print("\nRemoving highly correlated features...")

    cleaned_features, correlated_removed = (
        remove_highly_correlated_features(
            X_train_raw,
            cleaned_features,
            CORRELATION_THRESHOLD
        )
    )

    print(
        f"Highly correlated features removed: "
        f"{len(correlated_removed)}"
    )

    print(
        f"Features remaining: "
        f"{len(cleaned_features)}"
    )

    if len(cleaned_features) == 0:
        raise ValueError(
            f"No features remain after cleaning "
            f"for {experiment_name}."
        )

    # --------------------------------------------------------
    # MUTUAL INFORMATION
    # --------------------------------------------------------

    print("\nCalculating mutual information...")

    mi_df = calculate_mutual_information(
        X_train_raw,
        train_y,
        cleaned_features
    )

    mi_df.to_csv(
        os.path.join(
            experiment_dir,
            "mutual_information_ranking.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SELECT TOP FEATURES
    # --------------------------------------------------------

    selected_features = select_features_by_mi(
        mi_df,
        TARGET_FEATURES
    )

    print(
        f"\nTarget feature count: "
        f"{TARGET_FEATURES}"
    )

    print(
        f"Selected feature count: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # FEATURE GROUP COUNTS
    # --------------------------------------------------------

    selected_group_counts = {}

    for group in ["hc", "cnn", "meta"]:

        selected_group_counts[group] = sum(
            identify_feature_group(feature) == group
            for feature in selected_features
        )

    print("\nSelected feature groups:")

    print(
        f"  Handcrafted: "
        f"{selected_group_counts['hc']}"
    )

    print(
        f"  CNN: "
        f"{selected_group_counts['cnn']}"
    )

    print(
        f"  Metadata: "
        f"{selected_group_counts['meta']}"
    )

    # --------------------------------------------------------
    # SAVE FEATURE INFORMATION
    # --------------------------------------------------------

    save_feature_list(
        experiment_name,
        selected_features,
        candidate_features,
        constant_removed,
        correlated_removed
    )

    # --------------------------------------------------------
    # PREPARE SELECTED MATRICES
    # --------------------------------------------------------

    X_train_selected = train_df[
        selected_features
    ].copy()

    X_validation_selected = validation_df[
        selected_features
    ].copy()

    X_test_selected = test_df[
        selected_features
    ].copy()

    # --------------------------------------------------------
    # VALIDATION TRAINING
    # --------------------------------------------------------

    print_header(
        f"{experiment_name} - VALIDATION MODEL"
    )

    print(
        "Scaler will be fitted on TRAINING only."
    )

    print(
        "Validation will NOT be used for fitting."
    )

    validation_model, validation_scaler, validation_metrics = (
        train_validation_model(
            X_train_selected,
            train_y,
            X_validation_selected,
            validation_y
        )
    )

    print("\nValidation results:")

    for metric, value in validation_metrics.items():
        print(
            f"{metric.capitalize():<10}: "
            f"{value:.4f}"
        )

    validation_predictions = validation_model.predict(
        validation_scaler.transform(
            X_validation_selected
        )
    )

    validation_probability = validation_model.predict_proba(
        validation_scaler.transform(
            X_validation_selected
        )
    )[:, 1]

    validation_cm = confusion_matrix(
        validation_y,
        validation_predictions
    )

    save_confusion_matrix(
        validation_cm,
        experiment_name,
        "validation"
    )

    save_classification_report(
        validation_y,
        validation_predictions,
        experiment_name,
        "validation"
    )

    validation_prediction_df = pd.DataFrame({
        "media_id": validation_df["media_id"],
        "true_label": validation_df["label"],
        "predicted_label": np.where(
            validation_predictions == 1,
            "fake",
            "real"
        ),
        "fake_probability": validation_probability
    })

    validation_prediction_df.to_csv(
        os.path.join(
            experiment_dir,
            "validation_predictions.csv"
        ),
        index=False
    )

    validation_result = {
        "experiment": experiment_name,
        "feature_groups": "+".join(groups),
        "candidate_features": len(candidate_features),
        "constant_removed": len(constant_removed),
        "correlated_removed": len(correlated_removed),
        "features_after_cleaning": len(cleaned_features),
        "selected_features": len(selected_features),
        "handcrafted_selected": selected_group_counts["hc"],
        "cnn_selected": selected_group_counts["cnn"],
        "metadata_selected": selected_group_counts["meta"],
        **validation_metrics
    }

    all_validation_results.append(
        validation_result
    )

    # --------------------------------------------------------
    # FINAL TRAINING + TEST
    # --------------------------------------------------------

    print_header(
        f"{experiment_name} - FINAL TEST"
    )

    print(
        "Combining TRAIN + VALIDATION..."
    )

    print(
        "Fitting final scaler on TRAIN + VALIDATION..."
    )

    print(
        "TEST remains completely untouched."
    )

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
        test_y
    )

    print("\nFinal TEST results:")

    for metric, value in test_metrics.items():
        print(
            f"{metric.capitalize():<10}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        final_model,
        os.path.join(
            experiment_dir,
            "final_logistic_regression.joblib"
        )
    )

    joblib.dump(
        final_scaler,
        os.path.join(
            experiment_dir,
            "final_scaler.joblib"
        )
    )

    # --------------------------------------------------------
    # TEST CONFUSION MATRIX
    # --------------------------------------------------------

    test_cm = confusion_matrix(
        test_y,
        test_predictions
    )

    save_confusion_matrix(
        test_cm,
        experiment_name,
        "test"
    )

    save_classification_report(
        test_y,
        test_predictions,
        experiment_name,
        "test"
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_prediction_df = pd.DataFrame({
        "media_id": test_df["media_id"],
        "true_label": test_df["label"],
        "predicted_label": np.where(
            test_predictions == 1,
            "fake",
            "real"
        ),
        "fake_probability": test_probability
    })

    test_prediction_df.to_csv(
        os.path.join(
            experiment_dir,
            "test_predictions.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SAVE TEST METRICS
    # --------------------------------------------------------

    with open(
        os.path.join(
            experiment_dir,
            "test_metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=4
        )

    test_result = {
        "experiment": experiment_name,
        "feature_groups": "+".join(groups),
        "candidate_features": len(candidate_features),
        "constant_removed": len(constant_removed),
        "correlated_removed": len(correlated_removed),
        "features_after_cleaning": len(cleaned_features),
        "selected_features": len(selected_features),
        "handcrafted_selected": selected_group_counts["hc"],
        "cnn_selected": selected_group_counts["cnn"],
        "metadata_selected": selected_group_counts["meta"],
        **test_metrics
    }

    all_test_results.append(
        test_result
    )

    # --------------------------------------------------------
    # ROC CURVE DATA
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        test_y,
        test_probability
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )

    roc_curves.append({
        "experiment": experiment_name,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_value
    })

    # --------------------------------------------------------
    # PRECISION-RECALL DATA
    # --------------------------------------------------------

    precision, recall, _ = precision_recall_curve(
        test_y,
        test_probability
    )

    pr_auc_value = auc(
        recall,
        precision
    )

    pr_curves.append({
        "experiment": experiment_name,
        "precision": precision,
        "recall": recall,
        "auc": pr_auc_value
    })


# ============================================================
# SAVE COMPARISON TABLES
# ============================================================

print_header(
    "CREATING ABLATION COMPARISON"
)

validation_results_df = pd.DataFrame(
    all_validation_results
)

test_results_df = pd.DataFrame(
    all_test_results
)

validation_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ablation_validation_results.csv"
    ),
    index=False
)

test_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ablation_test_results.csv"
    ),
    index=False
)

print("\nValidation comparison:")
print(
    validation_results_df[
        [
            "experiment",
            "selected_features",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]
    ].to_string(index=False)
)

print("\nTEST comparison:")
print(
    test_results_df[
        [
            "experiment",
            "selected_features",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]
    ].to_string(index=False)
)


# ============================================================
# DETERMINE BEST VALIDATION EXPERIMENT
# ============================================================

if len(validation_results_df) > 0:

    best_validation = validation_results_df.sort_values(
        by=["roc_auc", "f1_score"],
        ascending=False
    ).iloc[0]

    print_header(
        "BEST FEATURE CONFIGURATION BASED ON VALIDATION"
    )

    print(
        f"Best experiment: "
        f"{best_validation['experiment']}"
    )

    print(
        f"Validation Accuracy: "
        f"{best_validation['accuracy']:.4f}"
    )

    print(
        f"Validation F1-score: "
        f"{best_validation['f1_score']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best_validation['roc_auc']:.4f}"
    )


# ============================================================
# ROC CURVE
# ============================================================

print_header(
    "CREATING ROC CURVE"
)

plt.figure(figsize=(8, 6))

for curve in roc_curves:

    plt.plot(
        curve["fpr"],
        curve["tpr"],
        label=(
            f"{curve['experiment']} "
            f"(AUC={curve['auc']:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Ablation Study - ROC Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_roc_curves.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

print_header(
    "CREATING PRECISION-RECALL CURVE"
)

plt.figure(figsize=(8, 6))

for curve in pr_curves:

    plt.plot(
        curve["recall"],
        curve["precision"],
        label=(
            f"{curve['experiment']} "
            f"(AUC={curve['auc']:.3f})"
        )
    )

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Ablation Study - Precision-Recall Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_precision_recall_curves.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# METRIC COMPARISON PLOT
# ============================================================

print_header(
    "CREATING METRIC COMPARISON PLOT"
)

if len(test_results_df) > 0:

    metrics_to_plot = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    x = np.arange(
        len(test_results_df)
    )

    width = 0.15

    plt.figure(
        figsize=(12, 7)
    )

    for i, metric in enumerate(metrics_to_plot):

        plt.bar(
            x + i * width,
            test_results_df[metric],
            width,
            label=metric.replace("_", " ").title()
        )

    plt.xticks(
        x + width * 2,
        test_results_df["experiment"],
        rotation=25,
        ha="right"
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Ablation Study - Test Performance"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "ablation_metric_comparison.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# METADATA CONTRIBUTION ANALYSIS
# ============================================================

print_header(
    "METADATA CONTRIBUTION ANALYSIS"
)

if len(test_results_df) > 0:

    lookup = test_results_df.set_index(
        "experiment"
    )

    contribution_rows = []

    if (
        "E4_Handcrafted_CNN" in lookup.index
        and "E5_All" in lookup.index
    ):

        base = lookup.loc[
            "E4_Handcrafted_CNN"
        ]

        combined = lookup.loc[
            "E5_All"
        ]

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]:

            contribution_rows.append({
                "comparison": "HC+CNN vs HC+CNN+Metadata",
                "metric": metric,
                "without_metadata": base[metric],
                "with_metadata": combined[metric],
                "change": combined[metric] - base[metric]
            })

    metadata_contribution_df = pd.DataFrame(
        contribution_rows
    )

    metadata_contribution_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "metadata_contribution.csv"
        ),
        index=False
    )

    if len(metadata_contribution_df) > 0:

        print(
            metadata_contribution_df.to_string(
                index=False
            )
        )


# ============================================================
# SAVE SUMMARY REPORT
# ============================================================

print_header(
    "SAVING ABLATION REPORT"
)

report_path = os.path.join(
    OUTPUT_DIR,
    "ablation_study_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ABLATION STUDY REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Deepfake Detection - Pilot Dataset\n\n"
    )

    f.write(
        "Experimental design:\n"
    )

    f.write(
        "- Train: 70%\n"
    )

    f.write(
        "- Validation: 15%\n"
    )

    f.write(
        "- Test: 15%\n"
    )

    f.write(
        "- Random state: 42\n"
    )

    f.write(
        "- Classifier: Logistic Regression\n"
    )

    f.write(
        "- Target selected features per experiment: 160\n"
    )

    f.write(
        "- Correlation threshold: 0.95\n\n"
    )

    f.write(
        "Leakage control:\n"
    )

    f.write(
        "- Feature selection performed using training data only.\n"
    )

    f.write(
        "- Validation data was not used during feature selection.\n"
    )

    f.write(
        "- Test data was not used during feature selection.\n"
    )

    f.write(
        "- Scaler was fitted only on training data during validation.\n"
    )

    f.write(
        "- Final scaler was fitted on training + validation data.\n"
    )

    f.write(
        "- Test data was used only for final evaluation.\n\n"
    )

    f.write(
        "VALIDATION RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    if len(validation_results_df) > 0:

        f.write(
            validation_results_df[
                [
                    "experiment",
                    "selected_features",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
            ].to_string(index=False)
        )

        f.write("\n\n")

    f.write(
        "FINAL TEST RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    if len(test_results_df) > 0:

        f.write(
            test_results_df[
                [
                    "experiment",
                    "selected_features",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
            ].to_string(index=False)
        )

        f.write("\n\n")

    if len(validation_results_df) > 0:

        f.write(
            "BEST VALIDATION CONFIGURATION\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Experiment: "
            f"{best_validation['experiment']}\n"
        )

        f.write(
            f"Accuracy: "
            f"{best_validation['accuracy']:.4f}\n"
        )

        f.write(
            f"F1-score: "
            f"{best_validation['f1_score']:.4f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{best_validation['roc_auc']:.4f}\n"
        )


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "random_state": RANDOM_STATE,
    "train_size": TRAIN_SIZE,
    "validation_size": VALIDATION_SIZE,
    "test_size": TEST_SIZE,
    "target_features": TARGET_FEATURES,
    "correlation_threshold": CORRELATION_THRESHOLD,
    "classifier": "LogisticRegression",
    "experiments": EXPERIMENTS
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "ablation_configuration.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# COMPLETED
# ============================================================

print_header(
    "ABLATION STUDY COMPLETED"
)

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print("\nMain files:")

print(
    "  ✓ ablation_validation_results.csv"
)

print(
    "  ✓ ablation_test_results.csv"
)

print(
    "  ✓ metadata_contribution.csv"
)

print(
    "  ✓ ablation_roc_curves.png"
)

print(
    "  ✓ ablation_precision_recall_curves.png"
)

print(
    "  ✓ ablation_metric_comparison.png"
)

print(
    "  ✓ ablation_study_report.txt"
)

print(
    "\nEach experiment also has its own directory "
    "containing selected features, models, predictions, "
    "confusion matrices and reports."
)

print(
    "\n✓ Feature selection was performed separately "
    "for every feature-group experiment."
)

print(
    "✓ Test data was kept separate until final evaluation."
)

print(
    "\nNext stage:"
)

print(
    "Interpret the ablation results and decide "
    "the final pipeline configuration for the full dataset."
)