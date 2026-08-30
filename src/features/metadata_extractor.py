"""
High-level metadata extraction from image and video files.

The metadata pipeline reads the original media files listed in
dataset_inventory.csv.

One row is produced for each media file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from features.metadata_utils import (
    get_file_info,
    get_media_type,
    get_image_properties,
    get_video_properties,
    get_exif_metadata,
    get_compression_info,
)



# Extract metadata from ONE media file
def extract_metadata_for_file(
    file_path: str | Path,
    label: str,
    media_id: str,
    dataset: str = "Unknown",
) -> dict:
    """
    Extract metadata from a single image or video.

    Image-specific and video-specific metadata are extracted
    according to the media type.
    """

    file_path = Path(file_path)

    # Basic information
    metadata = {
        "label": label,
        "media_id": media_id,
        "dataset": dataset,
    }

    
    # File information
    file_info = get_file_info(file_path)

    metadata.update(file_info)

    media_type = get_media_type(file_path)

    
    # Image
    if media_type == "image":

        metadata.update(
            get_image_properties(file_path)
        )

        metadata.update(
            get_exif_metadata(file_path)
        )

        # Video-only features
        metadata.update({
            "codec": "Unknown",
            "encoder": "Unknown",
            "duration": -1,
            "fps": -1,
        })

    
    # Video
    elif media_type == "video":

        metadata.update(
            get_video_properties(file_path)
        )

        # Videos normally do not contain image EXIF
        # information that we need for this pipeline.
        metadata.update({
            "has_exif": -1,
            "camera_make": "Unknown",
            "camera_model": "Unknown",
            "software": "Unknown",
            "has_gps": -1,
        })

    # Unknown
    else:

        metadata.update({
            "width": -1,
            "height": -1,
            "aspect_ratio": -1,
            "color_channels": -1,
            "bit_depth": -1,

            "codec": "Unknown",
            "encoder": "Unknown",

            "duration": -1,
            "fps": -1,

            "has_exif": -1,
            "camera_make": "Unknown",
            "camera_model": "Unknown",
            "software": "Unknown",
            "has_gps": -1,
        })


    # Compression / container information
    metadata.update(
        get_compression_info(file_path)
    )

    return metadata


# Extract metadata using dataset inventory

def extract_metadata_from_catalog(
    catalog_csv: str | Path,
    output_csv: str | Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Extract metadata from all media files listed in
    dataset_inventory.csv.

    Required columns in catalog:

        filepath
        dataset
        label
        video_name
    """

    catalog_csv = Path(catalog_csv)
    output_csv = Path(output_csv)

    print("=" * 60)
    print("Metadata Feature Extraction")
    print("=" * 60)

    print("\nReading dataset catalog:")
    print(catalog_csv)


    # Read catalog
    catalog_df = pd.read_csv(
        catalog_csv
    )

    print(
        f"Media files in catalog: {len(catalog_df)}"
    )

 
    # Optional limit
    if limit is not None:

        catalog_df = catalog_df.head(limit)

        print(
            f"Processing limited set: {len(catalog_df)}"
        )

    all_metadata = []


    # Process each media file
    for _, row in tqdm(
        catalog_df.iterrows(),
        total=len(catalog_df),
        desc="Extracting metadata"
    ):

        file_path = Path(
            str(row["filepath"]).strip()
        )

        label = str(
            row["label"]
        ).lower().strip()

        dataset = str(
            row["dataset"]
        ).strip()

        media_id = str(
            row["video_name"]
        ).strip()


        # Check file exists        
        if not file_path.exists():

            print(
                f"\nWARNING: File not found:"
            )

            print(file_path)

            continue

        try:

            metadata = extract_metadata_for_file(
                file_path=file_path,
                label=label,
                media_id=media_id,
                dataset=dataset,
            )

            all_metadata.append(
                metadata
            )

        except Exception as e:

            print(
                f"\nWARNING: Failed to process:"
            )

            print(file_path)

            print(
                f"Reason: {e}"
            )

    # Create DataFrame
    metadata_df = pd.DataFrame(
        all_metadata
    )

    # Save
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_df.to_csv(
        output_csv,
        index=False,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Metadata Extraction Summary")
    print("=" * 60)

    print(
        f"Files processed : {len(metadata_df)}"
    )

    print(
        f"Output columns  : {len(metadata_df.columns)}"
    )

    print(
        f"\nSaved metadata features:"
    )

    print(output_csv)

    print("=" * 60)

    return metadata_df