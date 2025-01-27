"""Top-level package for seegull."""

__author__ = """Bower"""
__email__ = "hello@getbower.com"
__version__ = "0.1.0"

from seegull.data import image

from seegull.data.image import Image, get_image_df, load_image, load_images
from seegull.models import dino, yolo
from seegull.models.dino import DINOv2, DINOv2TF, GroundingDINO

from seegull.models.yolo import YOLO, YOLOMultiLabel, YOLOValidation
from seegull.models.yolo import (
    format_training_data as format_yolo_training_data,
)

__all__ = [
    "image",
    "dino",
    "yolo",
    "Image",
    "get_image_df",
    "load_image",
    "load_images",
    "DINOv2",
    "DINOv2TF",
    "GroundingDINO",
    "YOLO",
    "YOLOMultiLabel",
    "YOLOValidation",
    "format_yolo_training_data",
]
