"""Compatibility entrypoint: prepare only Deepfakes.zip from Hugging Face.

For DVC runs, `prepare_dataset` calls `prepare_dataset_hf.py` directly.
This wrapper exists for older invocations of prepare_dataset.py.
"""
from prepare_dataset_hf import main

if __name__ == "__main__":
    main()
