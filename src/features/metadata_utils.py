"""
Utility functions for extracting metadata from image and video files.

Supports:
- File information
- Image properties
- Video properties
- EXIF metadata
- Compression information
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import json

from PIL import Image
from PIL.ExifTags import TAGS


# Media Type Detection
def get_media_type(file_path: str | Path) -> str:
    """
    Identify whether file is image or video.
    """

    extension = Path(file_path).suffix.lower()

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff"
    }

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }


    if extension in image_extensions:
        return "image"

    elif extension in video_extensions:
        return "video"

    return "unknown"



# File Information
def get_file_info(file_path: str | Path) -> dict:

    file_path = Path(file_path)

    return {

        "file_size": file_path.stat().st_size,

        "file_extension":
            file_path.suffix.lower(),

        "media_type":
            get_media_type(file_path)
    }



# Image Properties
def get_image_properties(file_path: str | Path) -> dict:
    """
    Extract image visual properties.
    """

    defaults = {
        "width": -1,
        "height": -1,
        "aspect_ratio": -1,
        "color_channels": -1,
        "bit_depth": -1
    }


    try:

        with Image.open(file_path) as img:

            width, height = img.size


            channel_map = {

                "1": 1,
                "L": 1,
                "P": 1,
                "RGB": 3,
                "RGBA": 4,
                "CMYK": 4

            }


            return {

                "width": width,

                "height": height,

                "aspect_ratio":
                    round(width / height, 4)
                    if height else -1,

                "color_channels":
                    channel_map.get(
                        img.mode,
                        -1
                    ),

                "bit_depth": 8
            }


    except Exception:

        return defaults



# Video Properties
def get_video_properties(file_path: str | Path) -> dict:
    """
    Extract video metadata using ffprobe.
    """

    defaults = {

        "width": -1,
        "height": -1,
        "aspect_ratio": -1,

        "color_channels": -1,
        "bit_depth": -1,

        "codec": "Unknown",
        "encoder": "Unknown",

        "duration": -1,
        "fps": -1

    }


    try:

        command = [

            "ffprobe",

            "-v",
            "quiet",

            "-print_format",
            "json",

            "-show_streams",

            "-show_format",

            str(file_path)

        ]


        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )


        data = json.loads(
            result.stdout
        )


        streams = data.get(
            "streams",
            []
        )


        video_stream = None


        for stream in streams:

            if stream.get("codec_type") == "video":

                video_stream = stream
                break



        if video_stream:

            width = video_stream.get(
                "width",
                -1
            )

            height = video_stream.get(
                "height",
                -1
            )


            fps_value = (
                video_stream
                .get(
                    "avg_frame_rate",
                    "0/1"
                )
            )


            num, den = fps_value.split("/")


            fps = (
                float(num) / float(den)
                if float(den) != 0
                else -1
            )


            return {

                "width": width,

                "height": height,

                "aspect_ratio":
                    round(
                        width / height,
                        4
                    )
                    if height else -1,

                "color_channels": -1,

                "bit_depth": -1,


                "codec":
                    video_stream.get(
                        "codec_name",
                        "Unknown"
                    ),


                "encoder":
                    data.get(
                        "format",
                        {}
                    ).get(
                        "tags",
                        {}
                    ).get(
                        "encoder",
                        "Unknown"
                    ),


                "duration":
                    float(
                        data
                        .get(
                            "format",
                            {}
                        )
                        .get(
                            "duration",
                            -1
                        )
                    ),


                "fps":
                    round(
                        fps,
                        2
                    )

            }


    except Exception:

        pass


    return defaults



# EXIF Metadata
def get_exif_metadata(file_path: str | Path) -> dict:

    metadata = {

        "has_exif": 0,

        "camera_make":
            "Unknown",

        "camera_model":
            "Unknown",

        "software":
            "Unknown",

        "has_gps": 0
    }


    try:

        with Image.open(file_path) as img:

            exif = img.getexif()


            if not exif:
                return metadata



            metadata["has_exif"] = 1



            for tag_id, value in exif.items():

                tag = TAGS.get(
                    tag_id,
                    tag_id
                )


                if tag == "Make":

                    metadata["camera_make"] = str(value)


                elif tag == "Model":

                    metadata["camera_model"] = str(value)


                elif tag == "Software":

                    metadata["software"] = str(value)


                elif tag == "GPSInfo":

                    metadata["has_gps"] = 1


    except Exception:

        pass


    return metadata


# Compression Information
def get_compression_info(file_path: str | Path) -> dict:

    extension = (
        Path(file_path)
        .suffix
        .replace(".", "")
        .upper()
    )

    return {
        "container_format": extension
    }