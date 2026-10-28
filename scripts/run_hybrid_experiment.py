"""Run one self-contained hybrid deepfake-detection experiment.

Everything comes from one experiment YAML (config/experiments/*.yaml):

    python scripts/run_hybrid_experiment.py --config config/experiments/hybrid_pilot_001.yaml --resume

Steps, each written under experiments/<experiment_id>/artifacts/:

1. catalog   videos per dataset and class, capped by max_videos_per_class
2. splits    stratified train/validation/test split by video
3. videos    frames -> full/face/expanded regions -> handcrafted features
4. cnn       pooled ImageNet CNN features for each region (GPU when available)
5. metadata  container metadata, preprocessed on TRAIN only
6. models    one random forest per feature combination (the ablation study's
             five: metadata, handcrafted, CNN, handcrafted+CNN, all), chosen
             on VALIDATION and reported on TEST

Results go to results/<experiment_id>/: metrics.json, comparison.csv and
run_status.json (written after every step, so a failed run says where).
With --resume, finished steps and already-processed videos are reused if the
config is unchanged; otherwise the artifacts are rebuilt from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from features.extract_color import create_zero_color_features, extract_color_features
from features.extract_dct import create_zero_dct_features, extract_dct_features
from features.extract_edge import create_zero_edge_features, extract_edge_features
from features.extract_glcm import create_zero_glcm_features, extract_glcm_features
from features.extract_lbp import create_zero_lbp_features, extract_lbp_features
from preprocessing.detect_faces import get_haar_face_detector
from preprocessing.extract_frames import extract_frames
from preprocessing.multi_region_extraction import extract_multi_regions_from_frame
from run_hybrid_metadata import extract_raw_metadata, preprocess_raw_metadata

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
LABELS = ("real", "fake")
REGIONS = ("full", "face", "expanded")
# The same five combinations as scripts/run_ablation_study.py.
COMBINATIONS = {
    "E1_Metadata": ("meta",),
    "E2_Handcrafted": ("hc",),
    "E3_CNN": ("cnn",),
    "E4_Handcrafted_CNN": ("hc", "cnn"),
    "E5_All": ("hc", "cnn", "meta"),
}
HANDCRAFTED = {
    "lbp": (extract_lbp_features, create_zero_lbp_features, "gray"),
    "glcm": (extract_glcm_features, create_zero_glcm_features, "gray"),
    "dct": (extract_dct_features, create_zero_dct_features, "gray"),
    "color": (extract_color_features, create_zero_color_features, "rgb"),
    "edge": (extract_edge_features, create_zero_edge_features, "gray"),
}
STEPS = ("catalog", "splits", "videos", "cnn", "metadata", "models")


# ---------------------------------------------------------------- helpers

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def config_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def media_key(media_id: str) -> str:
    """Readable, filesystem-safe and unique folder name for one video."""
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", media_id)[-80:]
    return f"{readable}_{hashlib.sha1(media_id.encode()).hexdigest()[:8]}"


class RunStatus:
    """results/<id>/run_status.json, rewritten after every step."""

    def __init__(self, path: Path, experiment_id: str):
        self.path = path
        self.recomputed = False  # once a step runs, every later step must run too
        self.state = {"experiment_id": experiment_id, "status": "running",
                      "started": now(), "finished": None, "error": None,
                      "steps": {step: {"status": "pending"} for step in STEPS}}
        self.save()

    def save(self) -> None:
        write_json(self.path, self.state)

    def step(self, name: str, status: str, **details) -> None:
        self.state["steps"][name] = {"status": status, **details}
        self.save()

    def finish(self, status: str, error: str | None = None) -> None:
        self.state.update(status=status, finished=now(), error=error)
        self.save()


def run_step(status: RunStatus, name: str, finished, resume: bool, func):
    """Run func() unless --resume finds the step finished; record the outcome.

    `finished` is a path the step writes last, or a callable returning a bool.
    """
    is_finished = finished() if callable(finished) else finished.exists()
    if resume and not status.recomputed and is_finished:
        print(f"\n== {name}: reusing previous output")
        status.step(name, "reused")
        return
    print(f"\n== {name}")
    status.recomputed = True
    started = time.time()
    status.step(name, "running")
    details = func() or {}
    status.step(name, "done", seconds=round(time.time() - started, 1), **details)


# ---------------------------------------------------------------- 1. catalog

def find_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def build_catalog(config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(config["random_seed"])
    cap = config.get("max_videos_per_class")
    rows = []
    for dataset in config["datasets"]:
        root = Path(dataset["root"])
        for label in LABELS:
            # Every listed alias folder that exists contributes its videos. On
            # case-insensitive disks (Windows, macOS) "Real" and "real" are one
            # folder, so keep each physical folder once.
            folders, seen = [], set()
            for name in dataset["folders"][label]:
                folder = root / name
                if folder.is_dir() and (identity := (folder.stat().st_dev, folder.stat().st_ino)) not in seen:
                    seen.add(identity)
                    folders.append(folder)
            videos = sorted({video for folder in folders for video in find_videos(folder)})
            if not videos:
                raise FileNotFoundError(
                    f"No {label} videos for {dataset['name']} under {root} "
                    f"(looked in {dataset['folders'][label]})")
            if cap and len(videos) > cap:
                picked = rng.choice(len(videos), size=cap, replace=False)
                videos = [videos[i] for i in sorted(picked)]
            for video in videos:
                folder = next(f for f in folders if f in video.parents)
                stem = video.relative_to(folder).with_suffix("").as_posix()
                media_id = f"{dataset['name']}/{label}/{stem}"
                rows.append({"media_id": media_id, "key": media_key(media_id),
                             "dataset": dataset["name"], "label": label,
                             "filepath": video.as_posix()})
    catalog = pd.DataFrame(rows)
    if catalog["media_id"].duplicated().any():
        raise ValueError("Duplicate media IDs in the catalog")
    return catalog


# ---------------------------------------------------------------- 2. splits

def split_catalog(catalog: pd.DataFrame, config: dict) -> pd.DataFrame:
    seed = config["random_seed"]
    val_frac, test_frac = config["val_fraction"], config["test_fraction"]
    holdout = val_frac + test_frac

    def split(df, size, strata):
        try:
            return train_test_split(df, test_size=size, random_state=seed, stratify=df[strata])
        except ValueError:
            # Too few videos per dataset/class stratum (tiny pilots): stratify by class only.
            return train_test_split(df, test_size=size, random_state=seed, stratify=df["label"])

    df = catalog.assign(stratum=catalog["dataset"] + "/" + catalog["label"])
    train, rest = split(df, holdout, "stratum")
    validation, test = split(rest, test_frac / holdout, "stratum")
    parts = [part.assign(split=name) for name, part in
             (("train", train), ("validation", validation), ("test", test))]
    return pd.concat(parts)[["media_id", "dataset", "label", "split"]].sort_values("media_id")


# ---------------------------------------------------------------- 3. videos

_FACE_DETECTOR = None


def handcrafted_region_features(path, prefix: str, families: list[str], size: int) -> dict:
    """The enabled handcrafted feature families for one region image."""
    image = cv2.imread(str(path)) if path else None
    features = {}
    if image is None:
        for name in families:
            features.update(HANDCRAFTED[name][1](prefix))
        features[f"{prefix}_missing"] = 1
        return features
    image = cv2.resize(image, (size, size))
    views = {"gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
             "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB)}
    for name in families:
        extract, _, view = HANDCRAFTED[name]
        features.update(extract(views[view], prefix))
    features[f"{prefix}_missing"] = 0
    return features


def process_video(video: dict, config: dict, artifacts: Path) -> dict:
    """Frames, regions and video-level handcrafted features for one video.

    The result is cached as videos/<key>.json, so --resume skips the video.
    """
    global _FACE_DETECTOR
    cache = artifacts / "videos" / f"{video['key']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = get_haar_face_detector()

    size = config["region_size"]
    frame_dir = artifacts / "tmp_frames" / video["key"]
    region_dir = artifacts / "regions" / video["key"]
    shutil.rmtree(frame_dir, ignore_errors=True)
    shutil.rmtree(region_dir, ignore_errors=True)
    for region in REGIONS:
        (region_dir / region).mkdir(parents=True, exist_ok=True)

    log_rows: list[dict] = []
    if extract_frames(video["filepath"], frame_dir, num_frames=config["frames_per_video"]):
        for frame_path in sorted(frame_dir.glob("*.jpg")):
            extract_multi_regions_from_frame(
                frame_path, region_dir, frame_path.name, video["label"], video["key"],
                log_rows, face_detector=_FACE_DETECTOR, size=(size, size),
                expanded_scale=config["expanded_face_scale"])
    shutil.rmtree(frame_dir, ignore_errors=True)

    families = [name for name, enabled in config["handcrafted_features"].items() if enabled]
    frames = []
    for row in log_rows:
        frame_features = {}
        for region in REGIONS:
            path = row.get(f"{region}_path")
            frame_features.update(handcrafted_region_features(
                path if isinstance(path, str) else None, region, families, size))
        frames.append(frame_features)

    result = {
        "media_id": video["media_id"],
        "frames": len(log_rows),
        "faces": int(sum(row.get("face_detected", 0) for row in log_rows)),
        "regions": {region: [row[f"{region}_path"] for row in log_rows if isinstance(row.get(f"{region}_path"), str)]
                    for region in REGIONS},
        "handcrafted": {},
    }
    if frames:
        table = pd.DataFrame(frames)
        # Mean and spread over frames, as in aggregate_handcrafted_features.py.
        for column in table.columns:
            result["handcrafted"][f"hc_{column}_mean"] = float(table[column].mean())
            result["handcrafted"][f"hc_{column}_std"] = float(table[column].std(ddof=0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


def process_videos(catalog: pd.DataFrame, config: dict, artifacts: Path, workers: int) -> dict:
    records = catalog.to_dict("records")
    jobs = Parallel(n_jobs=workers, return_as="generator")(
        delayed(process_video)(video, config, artifacts) for video in records)
    results = list(tqdm(jobs, total=len(records), desc="Videos", mininterval=10))
    shutil.rmtree(artifacts / "tmp_frames", ignore_errors=True)
    unreadable = [r["media_id"] for r in results if r["frames"] == 0]
    if len(unreadable) == len(results):
        raise RuntimeError("No frames could be read from any video")
    handcrafted = pd.DataFrame([{"media_id": r["media_id"], **r["handcrafted"]} for r in results])
    handcrafted.to_csv(artifacts / "handcrafted_features.csv", index=False)
    frames = sum(r["frames"] for r in results)
    faces = sum(r["faces"] for r in results)
    if unreadable:
        print(f"[WARN] {len(unreadable)} unreadable videos, e.g. {unreadable[:3]}")
    print(f"Frames: {frames}, frames with a face: {faces} ({faces / max(frames, 1):.1%})")
    return {"frames": frames, "frames_with_face": faces, "unreadable_videos": len(unreadable)}


# ---------------------------------------------------------------- 4. cnn

def extract_cnn(catalog: pd.DataFrame, config: dict, artifacts: Path) -> dict:
    from features.cnn_models import embed_images, get_cnn_model

    if config.get("cnn_pooling", "avg") != "avg":
        raise ValueError("Only cnn_pooling: avg is supported")
    model = get_cnn_model(config["cnn_model"], weights=config.get("cnn_weights", "imagenet"))
    device = next(model.parameters()).device
    print(f"CNN: {config['cnn_model']} on {device}")
    batch_size = config["cnn_batch_size"]
    size = config["region_size"]

    videos = {}
    for key in catalog["key"]:
        cached = json.loads((artifacts / "videos" / f"{key}.json").read_text(encoding="utf-8"))
        videos[cached["media_id"]] = cached["regions"]

    columns = {}
    for region in config["cnn_regions"]:
        items = [(media_id, path) for media_id, regions in videos.items() for path in regions[region]]
        vectors = []
        for start in tqdm(range(0, len(items), batch_size), desc=f"CNN {region}", mininterval=10):
            batch = [cv2.cvtColor(cv2.resize(cv2.imread(path), (size, size)), cv2.COLOR_BGR2RGB)
                     for _, path in items[start:start + batch_size]]
            vectors.append(embed_images(model, batch))
        if not items:
            print(f"[WARN] No {region} regions in any video; the {region} CNN features are all missing")
            columns[region] = pd.DataFrame({f"cnn_{region}_missing": 1}, index=list(videos))
            continue
        dim = vectors[0].shape[1]
        per_frame = pd.DataFrame(np.concatenate(vectors))
        per_frame.insert(0, "media_id", [media_id for media_id, _ in items])
        grouped = per_frame.groupby("media_id")
        mean, std = grouped.mean(), grouped.std(ddof=0)
        mean.columns = [f"cnn_{region}_{i + 1}_mean" for i in range(dim)]
        std.columns = [f"cnn_{region}_{i + 1}_std" for i in range(dim)]
        region_table = pd.concat([mean, std], axis=1).reindex(list(videos))
        # A region missing from every frame of a video (no face found) is zero-filled and flagged.
        region_table[f"cnn_{region}_missing"] = region_table.iloc[:, 0].isna().astype(int)
        columns[region] = region_table.fillna(0.0)
    table = pd.concat(columns.values(), axis=1).rename_axis("media_id").reset_index()
    table.to_csv(artifacts / "cnn_features.csv", index=False)
    return {"device": str(device), "cnn_features": table.shape[1] - 1}


# ---------------------------------------------------------------- 5. metadata

def build_metadata(catalog: pd.DataFrame, splits: pd.DataFrame, config: dict, artifacts: Path) -> dict:
    use_ffprobe = config.get("metadata", {}).get("use_ffprobe", True)
    raw = extract_raw_metadata(catalog, use_ffprobe=use_ffprobe)
    raw.to_csv(artifacts / "metadata_raw.csv", index=False)
    train_ids = set(splits.loc[splits["split"] == "train", "media_id"])
    features, preprocessor = preprocess_raw_metadata(raw, train_ids)
    joblib.dump(preprocessor, artifacts / "metadata_preprocessor.joblib")
    features.to_csv(artifacts / "metadata_features.csv", index=False)
    return {"metadata_features": features.shape[1] - 1}


# ---------------------------------------------------------------- 6. models

def evaluate(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, predictions, probabilities


def train_combinations(config: dict, artifacts: Path, result_dir: Path, workers: int) -> dict:
    splits = pd.read_csv(artifacts / "splits.csv", dtype={"media_id": str})
    table = splits
    for name in ("handcrafted_features.csv", "cnn_features.csv", "metadata_features.csv"):
        table = table.merge(pd.read_csv(artifacts / name, dtype={"media_id": str}),
                            on="media_id", how="left", validate="one_to_one")
    y = table["label"].map({"real": 0, "fake": 1}).to_numpy()
    parts = {name: table["split"] == name for name in ("train", "validation", "test")}
    for name, mask in parts.items():
        if len(set(y[mask])) < 2:
            raise ValueError(f"The {name} split needs both real and fake videos")

    (artifacts / "models").mkdir(exist_ok=True)
    (artifacts / "predictions").mkdir(exist_ok=True)
    rows = []
    for combination, groups in COMBINATIONS.items():
        features = [c for c in table.columns if c.split("_", 1)[0] in groups]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=config["rf_trees"], random_state=config["random_seed"],
                class_weight="balanced", n_jobs=workers)),
        ])
        train = parts["train"]
        model.fit(table.loc[train, features], y[train])
        joblib.dump(model, artifacts / "models" / f"{combination}.joblib")
        row = {"combination": combination, "feature_groups": "+".join(groups), "features": len(features)}
        for split_name in ("validation", "test"):
            mask = parts[split_name]
            metrics, predictions, probabilities = evaluate(model, table.loc[mask, features], y[mask])
            row.update({f"{split_name}_{k}": v for k, v in metrics.items()})
            pd.DataFrame({
                "media_id": table.loc[mask, "media_id"], "dataset": table.loc[mask, "dataset"],
                "label": table.loc[mask, "label"],
                "predicted": np.where(predictions == 1, "fake", "real"),
                "fake_probability": probabilities,
            }).to_csv(artifacts / "predictions" / f"{combination}_{split_name}.csv", index=False)
        rows.append(row)
        print(f"{combination:20s} {len(features):6d} features  "
              f"val F1 {row['validation_f1']:.3f}  test F1 {row['test_f1']:.3f}  "
              f"test AUC {row['test_roc_auc']}")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(result_dir / "comparison.csv", index=False)
    # Selection uses VALIDATION only; ties go to the combination with fewer features.
    best = comparison.sort_values(["validation_f1", "features"], ascending=[False, True]).iloc[0]
    counts = splits.groupby(["split", "label"]).size()
    metrics = {
        "experiment_id": config["experiment_id"],
        "videos": {split_name: {label: int(counts.get((split_name, label), 0)) for label in LABELS}
                   for split_name in ("train", "validation", "test")},
        "best_combination": best["combination"],
        "selection_metric": "validation_f1",
        "best_validation_f1": float(best["validation_f1"]),
        "best_test_f1": float(best["test_f1"]),
        "best_test_accuracy": float(best["test_accuracy"]),
        "best_test_roc_auc": best["test_roc_auc"],
        "combinations": {r["combination"]: {"validation_f1": r["validation_f1"], "test_f1": r["test_f1"],
                                            "test_accuracy": r["test_accuracy"], "test_roc_auc": r["test_roc_auc"]}
                         for r in rows},
    }
    return metrics


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hybrid deepfake-detection experiment")
    parser.add_argument("--config", required=True, help="Experiment YAML, e.g. config/experiments/hybrid_pilot_001.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse finished steps and processed videos when the config is unchanged")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Parallel CPU workers (Cluster-CI jobs get 4 cores)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment_id = config["experiment_id"]
    artifacts = Path("experiments") / experiment_id / "artifacts"
    result_dir = Path("results") / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = config_fingerprint(config)
    fingerprint_file = artifacts / "config_fingerprint.txt"
    resume = args.resume and fingerprint_file.exists() and fingerprint_file.read_text().strip() == fingerprint
    if args.resume and not resume and artifacts.exists():
        print("Config changed since the last run (or no previous run): starting from scratch.")
    if not resume:
        shutil.rmtree(artifacts, ignore_errors=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    fingerprint_file.write_text(fingerprint)
    (artifacts / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    status = RunStatus(result_dir / "run_status.json", experiment_id)
    print(f"Experiment {experiment_id}: artifacts in {artifacts}, results in {result_dir}")
    try:
        def load(name):
            return pd.read_csv(artifacts / name, dtype={"media_id": str})

        def catalog_step():
            catalog = build_catalog(config)
            catalog.to_csv(artifacts / "catalog.csv", index=False)
            print(catalog.groupby(["dataset", "label"]).size().to_string())
            return {"videos": len(catalog)}

        def splits_step():
            splits = split_catalog(load("catalog.csv"), config)
            splits.to_csv(artifacts / "splits.csv", index=False)
            print(splits.groupby(["split", "label"]).size().to_string())
            return {}

        run_step(status, "catalog", artifacts / "catalog.csv", resume, catalog_step)
        run_step(status, "splits", artifacts / "splits.csv", resume, splits_step)
        def videos_finished():
            return (artifacts / "handcrafted_features.csv").exists() and all(
                (artifacts / "videos" / f"{key}.json").exists() for key in load("catalog.csv")["key"])

        run_step(status, "videos", videos_finished, resume,
                 lambda: process_videos(load("catalog.csv"), config, artifacts, args.workers))
        run_step(status, "cnn", artifacts / "cnn_features.csv", resume,
                 lambda: extract_cnn(load("catalog.csv"), config, artifacts))
        run_step(status, "metadata", artifacts / "metadata_features.csv", resume,
                 lambda: build_metadata(load("catalog.csv"), load("splits.csv"), config, artifacts))

        metrics = {}

        def models_step():
            metrics.update(train_combinations(config, artifacts, result_dir, args.workers))
            return {"best_combination": metrics["best_combination"]}

        # Always retrain: it is cheap, and metrics.json must describe this run.
        run_step(status, "models", result_dir / "metrics.json", False, models_step)
        write_json(result_dir / "metrics.json", metrics)
        status.finish("completed")
        print(f"\nBest on validation: {metrics['best_combination']} "
              f"(validation F1 {metrics['best_validation_f1']:.3f}, test F1 {metrics['best_test_f1']:.3f})")
    except Exception as exc:
        running = [name for name, step in status.state["steps"].items() if step["status"] == "running"]
        for name in running:
            status.step(name, "failed")
        status.finish("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
