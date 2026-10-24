from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

def load_image(image_path, size=224):
    image = cv2.imread(str(image_path))
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size))

def extract_metadata(image_path):
    parts = image_path.parts
    label, video_id, region = "", "", ""
    frame = image_path.name
    for i, part in enumerate(parts):
        part_lower = part.lower()
        if part_lower in ["real", "fake"]:
            label = part_lower
            if i + 1 < len(parts): video_id = parts[i + 1]
            if i + 2 < len(parts): region = parts[i + 2]
            break
    return label, video_id, region, frame

def process_and_save_batch(batch_images, batch_metadata, model, preprocess_func, output_csv, is_first_batch):
    """Helper function to predict and append a batch to disk."""
    batch_array = preprocess_func(np.array(batch_images))
    batch_features = model.predict(batch_array, verbose=0)
    
    batch_rows = []
    for meta, feature in zip(batch_metadata, batch_features):
        label, video_id, region, frame = meta
        row = {
            "label": label,
            "video_id": video_id,
            "region": region,
            "frame": frame
        }
        for i, value in enumerate(feature):
            row[f"cnn_{i+1}"] = float(value)
        batch_rows.append(row)
        
    # Write directly to disk
    batch_df = pd.DataFrame(batch_rows)
    batch_df.to_csv(output_csv, mode='a', header=is_first_batch, index=False)
    
    # Force memory cleanup
    del batch_rows, batch_df, batch_array, batch_features
    # Do not clear the active model's Keras session between batches.
    # Clearing here can disrupt performance/state; caller may clear after extraction.

def extract_features(model, image_paths, model_name, output_csv, image_size=224, batch_size=16):
    if model_name.lower() == "mobilenetv2":
        preprocess = mobilenet_preprocess
    elif model_name.lower() == "efficientnetb0":
        preprocess = efficientnet_preprocess
    else:
        raise ValueError("Unknown preprocessing method.")

    # Ensure output file is empty before starting
    if output_csv.exists():
        output_csv.unlink()

    batch_images = []
    batch_metadata = []
    is_first_batch = True

    for img_path in tqdm(image_paths, desc="Extracting CNN Features"):
        image = load_image(img_path, image_size)
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        batch_images.append(image)
        batch_metadata.append(extract_metadata(img_path))

        if len(batch_images) == batch_size:
            process_and_save_batch(
                batch_images, batch_metadata, model, preprocess, output_csv, is_first_batch
            )
            is_first_batch = False
            batch_images = []
            batch_metadata = []

    # Process any remaining images
    if batch_images:
        process_and_save_batch(
            batch_images, batch_metadata, model, preprocess, output_csv, is_first_batch
        )