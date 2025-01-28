from pathlib import Path

import pytest

from seegull import YOLO, format_yolo_training_data

import tensorflow as tf


def test_format_yolo_training_data(annotation_df):
    output_path = Path("/tmp/yolo_training_data")

    # We need at least two images per class so this will fail
    with pytest.raises(ValueError):
        format_yolo_training_data(
            annotation_df, output_path, overwrite=True, target="object"
        )

    # But it will work if we limit to classes with at least two images
    subdf = annotation_df[
        annotation_df["object"].isin(
            ["Battery", "Flexibles/Bag/Wrapper/Foil/Net"]
        )
    ]
    format_yolo_training_data(
        subdf,
        output_path,
        overwrite=True,
        target="object",
        test_size=0.5,
    )

    # Confirm with multilabel
    format_yolo_training_data(
        subdf,
        output_path,
        overwrite=True,
        target=["object", "material"],
        test_size=0.5,
    )


### Regular, single-label YOLO ###
# TODO: These are terrible tests. Right now they just make sure that the
# code runs. But not that it's _correct_.
@pytest.fixture
def yolov8n():
    return YOLO("yolov8n.pt")


def test_call_default_yolo(im, yolov8n):
    assert yolov8n(im.path) is not None


def test_validate_yolo(yolov8n, annotation_df):
    val = yolov8n.validate(df=annotation_df, split=None, target="object")
    val.metrics_per_class()
    val.display_discrepencies(limit=1)
    val.plot_image_by_id(2724008258589670712)
