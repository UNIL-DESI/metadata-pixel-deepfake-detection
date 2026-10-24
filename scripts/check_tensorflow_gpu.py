#!/usr/bin/env python3
"""Standalone GPU smoke test for the DVC Cluster-CI runner.

Only checks NVIDIA driver/GPU visibility and a TensorFlow MobileNetV2
forward pass. Never reads the dataset and never downloads ImageNet weights.
All details go into results/gpu_check/diagnostics.log and status.json.
"""
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("results/gpu_check")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "diagnostics.log"
STATUS = OUT / "status.json"


def write_log(message):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(str(message).rstrip() + "\n")


def main():
    LOG.write_text("", encoding="utf-8")
    report = {
        "status": "failed",
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "gpu_visible_to_tensorflow": False,
        "tensorflow_mobilenetv2_inference_ok": False,
    }
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False
        )
        report["nvidia_smi_returncode"] = probe.returncode
        report["nvidia_smi_result"] = probe.stdout.strip()
        write_log(f"nvidia-smi exit code: {probe.returncode}")
        write_log(probe.stdout)
        if probe.stderr:
            write_log(probe.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        write_log(f"nvidia-smi query failed: {exc}")

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        import tensorflow as tf

        report["tensorflow_version"] = tf.__version__
        write_log(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        report["tensorflow_gpu_devices"] = [str(gpu) for gpu in gpus]
        report["gpu_visible_to_tensorflow"] = bool(gpus)
        write_log(f"Physical GPUs: {report['tensorflow_gpu_devices']}")
        if not gpus:
            raise RuntimeError("No GPU visible to TensorFlow. Do not run the full pipeline.")

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                write_log(f"Memory growth not set: {exc}")

        tf.config.set_soft_device_placement(False)
        with tf.device("/GPU:0"):
            result = tf.linalg.matmul(tf.ones((128, 128)), tf.ones((128, 128)))
        report["matmul_device"] = result.device
        report["matmul_ok"] = result.shape == (128, 128)
        write_log(f"GPU matrix multiply: {result.device}, shape={result.shape}")
        if "GPU:0" not in result.device:
            raise RuntimeError("Matrix multiply did not execute on GPU:0")

        from tensorflow.keras.applications import MobileNetV2
        model = MobileNetV2(weights=None, include_top=False, pooling="avg",
                            input_shape=(224, 224, 3))
        with tf.device("/GPU:0"):
            output = model(tf.zeros((1, 224, 224, 3)), training=False)
        report["mobilenetv2_output_shape"] = [int(x) for x in output.shape]
        report["tensorflow_mobilenetv2_inference_ok"] = True
        report["status"] = "passed"
        write_log(f"MobileNetV2 inference passed, output shape={output.shape}")
    except Exception as exc:
        report["error"] = str(exc)
        write_log(traceback.format_exc())
    finally:
        STATUS.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        write_log(f"Final test result: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
