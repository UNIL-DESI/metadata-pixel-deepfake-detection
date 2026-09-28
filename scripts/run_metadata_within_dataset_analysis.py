import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu


ALIGNED_METADATA = "features/aligned/aligned_metadata.csv"
INVENTORY = "dataset_inventory.csv"
OUTPUT = "results/metadata_within_dataset_analysis.csv"


def main():

    print("=" * 90)
    print("WITHIN-DATASET METADATA REAL/FAKE ANALYSIS")
    print("=" * 90)

    meta = pd.read_csv(ALIGNED_METADATA)
    inventory = pd.read_csv(INVENTORY)

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

    exclude = {
        "media_id",
        "label",
        "dataset"
    }

    features = [
        c for c in df.columns
        if c not in exclude
        and df[c].nunique() > 1
    ]

    datasets = [
        "FF++",
        "Celeb-DF",
        "DFDC"
    ]

    results = []

    for feature in features:

        for dataset in datasets:

            subset = df[
                df["dataset"] == dataset
            ].copy()

            if feature not in subset.columns:
                continue

            # Numeric/one-hot features should already be numeric
            x = subset[feature]
            y = subset["label"].map({
                "real": 0,
                "fake": 1
            })

            # Skip constant-within-dataset features
            if x.nunique() <= 1:
                results.append({
                    "feature": feature,
                    "dataset": dataset,
                    "real_mean": x[subset["label"] == "real"].mean(),
                    "fake_mean": x[subset["label"] == "fake"].mean(),
                    "auc": 0.5,
                    "auc_signal_strength": 0.5,
                    "direction": "constant",
                    "mutual_information": 0.0,
                    "mannwhitney_p": np.nan
                })
                continue

            real_values = x[
                subset["label"] == "real"
            ].values

            fake_values = x[
                subset["label"] == "fake"
            ].values

            # --------------------------------------------
            # AUC
            # --------------------------------------------
            try:
                auc = roc_auc_score(
                    y,
                    x
                )

                # Signal strength ignores direction
                auc_strength = max(
                    auc,
                    1 - auc
                )

                if auc > 0.5:
                    direction = "higher_in_fake"
                elif auc < 0.5:
                    direction = "higher_in_real"
                else:
                    direction = "no_direction"

            except Exception:
                auc = np.nan
                auc_strength = np.nan
                direction = "unknown"

            # --------------------------------------------
            # Mutual information
            # --------------------------------------------
            try:
                mi = mutual_info_classif(
                    subset[[feature]],
                    y,
                    random_state=42
                )[0]
            except Exception:
                mi = np.nan

            # --------------------------------------------
            # Mann-Whitney
            # --------------------------------------------
            try:
                _, p = mannwhitneyu(
                    real_values,
                    fake_values,
                    alternative="two-sided"
                )
            except Exception:
                p = np.nan

            results.append({
                "feature": feature,
                "dataset": dataset,

                "real_mean": np.mean(real_values),
                "fake_mean": np.mean(fake_values),

                "auc": auc,
                "auc_signal_strength": auc_strength,

                "direction": direction,

                "mutual_information": mi,
                "mannwhitney_p": p
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT,
        index=False
    )

    print("\n" + "=" * 90)
    print("WITHIN-DATASET RESULTS")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Summary across datasets
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("CROSS-DATASET FEATURE CONSISTENCY")
    print("=" * 90)

    summaries = []

    for feature in features:

        f = results_df[
            results_df["feature"] == feature
        ]

        useful = f[
            f["auc_signal_strength"] >= 0.60
        ]

        directions = useful["direction"].unique()

        summaries.append({
            "feature": feature,

            "mean_auc_strength":
                f["auc_signal_strength"].mean(),

            "min_auc_strength":
                f["auc_signal_strength"].min(),

            "max_auc_strength":
                f["auc_signal_strength"].max(),

            "datasets_auc_ge_0.60":
                len(useful),

            "consistent_direction":
                len(directions) == 1
                and len(useful) >= 2
        })

    summary_df = pd.DataFrame(
        summaries
    ).sort_values(
        [
            "datasets_auc_ge_0.60",
            "mean_auc_strength"
        ],
        ascending=False
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    summary_df.to_csv(
        "results/metadata_within_dataset_summary.csv",
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)
    print(
        "results/metadata_within_dataset_summary.csv"
    )


if __name__ == "__main__":
    main()