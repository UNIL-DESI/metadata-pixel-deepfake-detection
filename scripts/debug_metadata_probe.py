from __future__ import annotations

import static_ffmpeg
static_ffmpeg.add_paths()  # This magically makes "ffprobe" available to the system

import subprocess
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import pandas as pd


CATALOG = "dataset_inventory.csv"
NUM_VIDEOS = 5


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_ffprobe(video_path):
    ffprobe = shutil.which("ffprobe")

    if ffprobe is None:
        return {
            "available": False,
            "error": "ffprobe was not found in PATH."
        }

    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    start = time.perf_counter()

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )

    elapsed = time.perf_counter() - start

    output = {
        "available": True,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stderr": result.stderr,
    }

    if result.returncode == 0:
        try:
            output["json"] = json.loads(result.stdout)
        except json.JSONDecodeError:
            output["json_error"] = "Could not decode ffprobe JSON."
            output["stdout"] = result.stdout[:2000]
    else:
        output["stdout"] = result.stdout[:2000]

    return output


def extract_ffprobe_values(result):
    values = {
        "width": None,
        "height": None,
        "fps": None,
        "duration": None,
        "codec": None,
        "encoder": None,
        "pixel_format": None,
        "bit_rate": None,
    }

    if "json" not in result:
        return values

    data = result["json"]

    video_stream = None

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream:
        values["width"] = video_stream.get("width")
        values["height"] = video_stream.get("height")
        values["codec"] = video_stream.get("codec_name")
        values["pixel_format"] = video_stream.get("pix_fmt")
        values["bit_rate"] = video_stream.get("bit_rate")

        fps_value = (
            video_stream.get("avg_frame_rate")
            or video_stream.get("r_frame_rate")
        )

        if fps_value and fps_value != "0/0":
            try:
                numerator, denominator = fps_value.split("/")
                denominator = float(denominator)

                if denominator != 0:
                    values["fps"] = float(numerator) / denominator
            except Exception:
                values["fps"] = fps_value

        values["duration"] = video_stream.get("duration")

        tags = video_stream.get("tags", {})
        values["encoder"] = tags.get("encoder")

    format_info = data.get("format", {})

    if values["duration"] is None:
        values["duration"] = format_info.get("duration")

    if values["encoder"] is None:
        values["encoder"] = format_info.get("tags", {}).get("encoder")

    return values


def test_opencv(video_path):
    start = time.perf_counter()

    cap = cv2.VideoCapture(video_path)

    opened = cap.isOpened()

    result = {
        "opened": opened,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration": None,
    }

    if opened:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        result["width"] = width
        result["height"] = height
        result["fps"] = fps
        result["frame_count"] = frame_count

        if fps and fps > 0:
            result["duration"] = frame_count / fps

    cap.release()

    result["elapsed_seconds"] = time.perf_counter() - start

    return result


def main():
    print_section("METADATA EXTRACTION DEBUG")

    print("Current working directory:")
    print(os.getcwd())

    print("\nCatalog:")
    print(Path(CATALOG).resolve())

    if not os.path.exists(CATALOG):
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG}"
        )

    print_section("ENVIRONMENT CHECK")

    ffprobe_path = shutil.which("ffprobe")

    print("ffprobe path:")
    print(ffprobe_path)

    if ffprobe_path:
        version = subprocess.run(
            [ffprobe_path, "-version"],
            capture_output=True,
            text=True,
        )

        print("\nffprobe version:")
        print(version.stdout.splitlines()[0])

    print("\nOpenCV version:")
    print(cv2.__version__)

    print_section("LOADING CATALOG")

    df = pd.read_csv(CATALOG)

    print("Catalog shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    if "filepath" not in df.columns:
        raise ValueError(
            "dataset_inventory.csv does not contain a 'filepath' column."
        )

    samples = df.head(NUM_VIDEOS)

    for index, row in samples.iterrows():

        print_section(f"VIDEO {index + 1}")

        video_path = str(row["filepath"])

        print("Dataset:")
        print(row.get("dataset"))

        print("\nLabel:")
        print(row.get("label"))

        print("\nCatalog filepath:")
        print(video_path)

        print("\nAbsolute/resolved path:")
        try:
            print(Path(video_path).resolve())
        except Exception:
            print("Could not resolve path.")

        exists = os.path.exists(video_path)

        print("\nExists:")
        print(exists)

        if not exists:
            print(
                "\n!!! FILE DOES NOT EXIST FROM INSIDE THIS CONTAINER !!!"
            )
            continue

        print("\nFile size:")
        print(os.path.getsize(video_path), "bytes")

        # ------------------------------------------------------------
        # FFPROBE TEST
        # ------------------------------------------------------------

        print("\n--- FFPROBE TEST ---")

        probe_result = run_ffprobe(video_path)

        print("ffprobe available:")
        print(probe_result.get("available"))

        print("return code:")
        print(probe_result.get("returncode"))

        print("elapsed:")
        print(probe_result.get("elapsed_seconds"))

        if probe_result.get("stderr"):
            print("\nffprobe stderr:")
            print(probe_result["stderr"][:2000])

        values = extract_ffprobe_values(probe_result)

        print("\nExtracted ffprobe values:")

        for key, value in values.items():
            print(f"{key:15}: {value}")

        # ------------------------------------------------------------
        # OPENCV TEST
        # ------------------------------------------------------------

        print("\n--- OPENCV TEST ---")

        cv_result = test_opencv(video_path)

        for key, value in cv_result.items():
            print(f"{key:15}: {value}")

    print_section("DEBUG RUN COMPLETED")


if __name__ == "__main__":
    main()