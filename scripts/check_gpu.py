#!/usr/bin/env python3
"""Standalone GPU smoke test for the DVC Cluster-CI runner.

Only checks NVIDIA driver/GPU visibility and a PyTorch MobileNetV2 forward
pass through the project's own feature extractor. Never reads the dataset
and never downloads ImageNet weights.
All details go into results/gpu_check/diagnostics.log and status.json.
"""
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

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
        "gpu_visible_to_torch": False,
        "torch_mobilenetv2_inference_ok": False,
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
        import numpy as np
        import torch
        from features.cnn_models import embed_images, get_cnn_model

        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        write_log(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
        report["gpu_visible_to_torch"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU visible to PyTorch. Do not run the full pipeline.")
        device = torch.device("cuda:0")
        report["torch_gpu_name"] = torch.cuda.get_device_name(device)
        write_log(f"GPU: {report['torch_gpu_name']}")

        result = torch.ones((128, 128), device=device) @ torch.ones((128, 128), device=device)
        report["matmul_device"] = str(result.device)
        report["matmul_ok"] = bool(result[0, 0].item() == 128.0)
        write_log(f"GPU matrix multiply: {result.device}, ok={report['matmul_ok']}")
        if not report["matmul_ok"]:
            raise RuntimeError("Matrix multiply on GPU returned a wrong result")

        model = get_cnn_model("mobilenetv2", weights=None, device=device)
        output = embed_images(model, np.zeros((2, 224, 224, 3), dtype=np.uint8))
        report["mobilenetv2_output_shape"] = [int(x) for x in output.shape]
        if output.shape != (2, 1280):
            raise RuntimeError(f"Unexpected MobileNetV2 output shape {output.shape}")
        report["torch_mobilenetv2_inference_ok"] = True
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
