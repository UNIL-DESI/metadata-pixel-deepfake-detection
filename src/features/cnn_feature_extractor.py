from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)


def load_image(image_path, size=224):
    """
    Load and preprocess a single image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (size, size)
    )

    return image


def extract_metadata(image_path):
    """
    Extract metadata from image path.

    Expected folder structure:

    MultiRegionFrames/
        real/
            video_id/
                full/
                    frame_000.jpg

    """

    parts = image_path.parts

    label = ""
    video_id = ""
    region = ""
    frame = image_path.name

    for i, part in enumerate(parts):

        part_lower = part.lower()

        if part_lower in ["real", "fake"]:

            label = part_lower

            if i + 1 < len(parts):
                video_id = parts[i + 1]

            if i + 2 < len(parts):
                region = parts[i + 2]

            break

    return label, video_id, region, frame


def extract_features(
        model,
        image_paths,
        model_name,
        image_size=224,
        batch_size=16
):
    """
    Extract CNN features and return
    a frame-level DataFrame.
    """

    if model_name.lower() == "mobilenetv2":

        preprocess = mobilenet_preprocess

    elif model_name.lower() == "efficientnetb0":

        preprocess = efficientnet_preprocess

    else:

        raise ValueError(
            "Unknown preprocessing method."
        )

    batch_images = []
    batch_metadata = []

    rows = []

    for img_path in tqdm(
            image_paths,
            desc="Extracting CNN Features"
    ):

        image = load_image(
            img_path,
            image_size
        )

        if image is None:

            print(
                f"Skipping unreadable image: {img_path}"
            )

            continue

        batch_images.append(image)

        batch_metadata.append(
            extract_metadata(img_path)
        )

        if len(batch_images) == batch_size:

            batch_array = preprocess(
                np.array(batch_images)
            )

            batch_features = model.predict(
                batch_array,
                verbose=0
            )

            for meta, feature in zip(
                    batch_metadata,
                    batch_features
            ):

                label, video_id, region, frame = meta

                row = {
                    "label": label,
                    "video_id": video_id,
                    "region": region,
                    "frame": frame
                }

                for i, value in enumerate(feature):

                    row[f"cnn_{i+1}"] = float(value)

                rows.append(row)

            batch_images = []
            batch_metadata = []

    # Process remaining images

    if batch_images:

        batch_array = preprocess(
            np.array(batch_images)
        )

        batch_features = model.predict(
            batch_array,
            verbose=0
        )

        for meta, feature in zip(
                batch_metadata,
                batch_features
        ):

            label, video_id, region, frame = meta

            row = {
                "label": label,
                "video_id": video_id,
                "region": region,
                "frame": frame
            }

            for i, value in enumerate(feature):

                row[f"cnn_{i+1}"] = float(value)

            rows.append(row)

    frame_df = pd.DataFrame(rows)

    return frame_df