"""Top-level package for seegull."""

__author__ = """Bower"""
__email__ = "hello@getbower.com"
__version__ = "0.1.0"

from seegull.data import image
from seegull.data.image import Image, get_image_df, load_image, load_images, load_images_from_df
from seegull.models import yolo
from seegull.models.yolo import YOLO, YOLOMultiLabel, YOLOValidation
from seegull.models.yolo import (
    format_training_data as format_yolo_training_data,
)

__all__ = [
    "image",
    "yolo",
    "Image",
    "get_image_df",
    "load_images_from_df",
    "load_image",
    "load_images",
    "YOLO",
    "YOLOMultiLabel",
    "YOLOValidation",
    "format_yolo_training_data",
]
