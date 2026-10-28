# metadata-pixel-deepfake-detection
A lightweight deepfake detection framework that combines metadata and pixel-level features for accurate video/image authenticity classification.

#Must activate the environment before running by ".\.venv\Scripts\Activate"

## Running an experiment

Each experiment is one YAML file in `config/experiments/` and one DVC stage. `hybrid_pilot_001` caps the data at 30 videos per class and dataset; `hybrid_001` uses everything. Start a new one with `python scripts/create_hybrid_experiment.py hybrid_002` and add a matching stage to `dvc.yaml`.

```bash
python scripts/run_hybrid_experiment.py --config config/experiments/hybrid_pilot_001.yaml --resume
```

It builds the catalog, splits by video, extracts regions, handcrafted, CNN (PyTorch, on the GPU when there is one) and metadata features, then trains one random forest per feature combination (metadata, handcrafted, CNN, handcrafted+CNN, all). The model is chosen on validation and reported on test. Results are in `results/<experiment_id>/` (`metrics.json`, `comparison.csv`, `run_status.json`), everything else in `experiments/<experiment_id>/artifacts/`.

The data comes from the `prepare_dataset` stage, which downloads `Deepfakes.zip` from the `dinithiKP/Deepfakes` Hugging Face dataset. If that dataset is private, add a read token as the `HF_TOKEN` repository secret.

## On the cluster

`.cluster-ci` picks the stage (`STAGES`) and the resources; `cluster-run` sends the working tree. Runs use `nvcr.io/nvidia/pytorch:25.06-py3` on `isipol09`, which already has torch, numpy, pandas and scikit-learn; `pyproject.toml` lists only what it lacks, and the `local` extra adds the rest for a laptop (`uv sync --extra local`). Don't cancel a run while it installs dependencies: that leaves the cluster's cached environment broken (`dvc-viewer: command not found` on every later run), and only a change of Docker image clears it.

## Tests

```bash
uv run --extra local --extra dev pytest tests
```

The tests build a small synthetic dataset, so they need no GPU, network or real data.
