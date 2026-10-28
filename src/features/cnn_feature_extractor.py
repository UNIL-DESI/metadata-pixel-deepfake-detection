from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

from features.cnn_models import embed_images

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

def process_and_save_batch(batch_images, batch_metadata, model, output_csv, is_first_batch):
    """Helper function to predict and append a batch to disk."""
    batch_features = embed_images(model, batch_images)
    
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
    del batch_rows, batch_df, batch_features

def extract_features(model, image_paths, model_name, output_csv, image_size=224, batch_size=16):
    """Write one row of pooled CNN features per image to output_csv.

    model_name is kept for the call signature; preprocessing is the same
    ImageNet normalisation for every torchvision backbone.
    """
    output_csv = Path(output_csv)
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
                batch_images, batch_metadata, model, output_csv, is_first_batch
            )
            is_first_batch = False
            batch_images = []
            batch_metadata = []

    # Process any remaining images
    if batch_images:
        process_and_save_batch(
            batch_images, batch_metadata, model, output_csv, is_first_batch
        )