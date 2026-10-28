"""Smoke tests on a tiny synthetic dataset; no GPU, network or real data needed.

    uv run --extra local --extra dev pytest tests
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from synthetic_dataset import build_zip  # noqa: E402


def run(script: str, *args: str, cwd: Path) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=cwd, check=True)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """data/raw/Hybrid_Deepfake_Dataset unpacked from a synthetic archive."""
    workdir = tmp_path_factory.mktemp("work")
    archive = build_zip(workdir / "Deepfakes.zip", videos_per_class=3)
    run("prepare_dataset_hf.py", "--local-zip", str(archive),
        "--dest", "data/raw/Hybrid_Deepfake_Dataset", cwd=workdir)
    return workdir


def test_cnn_backbones_return_pooled_vectors():
    from features.cnn_models import embed_images, get_cnn_model

    for name in ("mobilenetv2", "efficientnetb0"):
        model = get_cnn_model(name, weights=None)
        features = embed_images(model, np.zeros((2, 224, 224, 3), dtype=np.uint8))
        assert features.shape == (2, 1280)
        assert np.isfinite(features).all()


def test_prepare_dataset_unpacks_every_class(dataset: Path):
    summary = json.loads((dataset / "data/raw/Hybrid_Deepfake_Dataset/hf_extraction_summary.json").read_text())
    assert summary["counts"] == {f"{d}/{c}": 3 for d in ("FF++", "Celeb-DF", "DFDC") for c in ("Real", "Fake")}


def test_hybrid_experiment_runs_and_resumes(dataset: Path):
    config = yaml.safe_load((ROOT / "config/experiments/hybrid_pilot_001.yaml").read_text())
    # Small and offline: few frames, random CNN weights instead of an ImageNet download.
    config.update(experiment_id="hybrid_test", frames_per_video=3, cnn_weights=None, rf_trees=20)
    config_path = dataset / "hybrid_test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    run("run_hybrid_experiment.py", "--config", str(config_path), "--resume", "--workers", "2", cwd=dataset)
    results = dataset / "results/hybrid_test"
    status = json.loads((results / "run_status.json").read_text())
    assert status["status"] == "completed"
    assert all(step["status"] == "done" for step in status["steps"].values())

    comparison = pd.read_csv(results / "comparison.csv")
    assert list(comparison["combination"]) == [
        "E1_Metadata", "E2_Handcrafted", "E3_CNN", "E4_Handcrafted_CNN", "E5_All"]
    assert (comparison["features"] > 0).all()
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["best_combination"] in set(comparison["combination"])
    assert sum(sum(v.values()) for v in metrics["videos"].values()) == 18

    # A second run with --resume reuses every step except training.
    run("run_hybrid_experiment.py", "--config", str(config_path), "--resume", "--workers", "2", cwd=dataset)
    steps = json.loads((results / "run_status.json").read_text())["steps"]
    assert [s["status"] for s in steps.values()] == ["reused"] * 5 + ["done"]
