"""Container metadata features for a hybrid experiment.

Used by run_hybrid_experiment.py, and runnable on its own once that script has
written the experiment's catalog and splits:

    python scripts/run_hybrid_metadata.py --config config/experiments/hybrid_pilot_001.yaml

Raw metadata is read from each original video (ffprobe through static-ffmpeg,
or OpenCV when `metadata.use_ffprobe` is false). Preprocessing reuses
preprocess_metadata.py and is fitted on the TRAIN split only, so validation and
test statistics never leak into the features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from features.metadata_extractor import extract_metadata_for_file
from metadata.extract_metadata import extract_metadata_opencv
from preprocess_metadata import (
    build_preprocessor,
    clean_missing_values,
    get_feature_names,
    identify_feature_types,
    separate_target,
)

# Columns that describe where a file came from rather than how it was encoded.
IDENTIFIER_COLUMNS = ["video_path"]


def extract_raw_metadata(catalog: pd.DataFrame, use_ffprobe: bool = True) -> pd.DataFrame:
    """One row of raw metadata per video in the experiment catalog."""
    if use_ffprobe:
        import static_ffmpeg
        static_ffmpeg.add_paths()  # puts ffprobe on PATH

    rows = []
    for row in tqdm(catalog.itertuples(index=False), total=len(catalog),
                    desc="Extracting metadata", mininterval=10):
        if use_ffprobe:
            metadata = extract_metadata_for_file(
                row.filepath, label=row.label, media_id=row.media_id, dataset=row.dataset,
            )
        else:
            metadata = {"label": row.label, "media_id": row.media_id, "dataset": row.dataset}
            metadata.update(extract_metadata_opencv(row.filepath))
        rows.append(metadata)
    return pd.DataFrame(rows).drop(columns=IDENTIFIER_COLUMNS, errors="ignore")


def preprocess_raw_metadata(raw: pd.DataFrame, train_ids: set[str]) -> tuple[pd.DataFrame, object]:
    """Train-fitted metadata features, one `meta_` column each, keyed by media_id."""
    df = clean_missing_values(raw)
    X, _ = separate_target(df)
    # Columns that are empty for every video carry no information and break the imputers.
    X = X.dropna(axis=1, how="all")
    numerical, categorical = identify_feature_types(X)
    preprocessor = build_preprocessor(numerical, categorical)
    train_mask = df["media_id"].isin(train_ids)
    if not train_mask.any():
        raise ValueError("No training videos found in the metadata table.")
    preprocessor.fit(X.loc[train_mask])
    features = pd.DataFrame(
        preprocessor.transform(X),
        columns=[f"meta_{name}" for name in get_feature_names(preprocessor)],
    )
    features.insert(0, "media_id", df["media_id"].to_numpy())
    return features, preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and preprocess metadata for a hybrid experiment")
    parser.add_argument("--config", required=True, help="Experiment YAML, e.g. config/experiments/hybrid_pilot_001.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    artifacts = Path("experiments") / config["experiment_id"] / "artifacts"
    catalog = pd.read_csv(artifacts / "catalog.csv", dtype={"media_id": str})
    splits = pd.read_csv(artifacts / "splits.csv", dtype={"media_id": str})
    use_ffprobe = config.get("metadata", {}).get("use_ffprobe", True)

    raw = extract_raw_metadata(catalog, use_ffprobe=use_ffprobe)
    raw.to_csv(artifacts / "metadata_raw.csv", index=False)
    features, _ = preprocess_raw_metadata(raw, set(splits.loc[splits["split"] == "train", "media_id"]))
    features.to_csv(artifacts / "metadata_features.csv", index=False)
    print(f"Saved {features.shape[1] - 1} metadata features for {len(features)} videos to {artifacts}")


if __name__ == "__main__":
    main()
