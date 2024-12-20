import itertools
import json
import shutil
from functools import partial
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.python.types.core import ConcreteFunction
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from seegull.data.bounding_box import iou


class YOLO:
    """A wrapper class for ultralytics.YOLO model

    Adds functionality for validating, generating predictions and exporting a
    tensorflow version with a custom signature.
    """

    def __init__(self, model_path):
        # Import ultralytics locally and store the version so we don't
        # import both ultralytics and ultralytics_bower in one session.
        import ultralytics

        ultralytics.checks()
        self.ultralytics = "ultralytics"

        self.path = Path(model_path)
        self.model = ultralytics.YOLO(model_path)
        self.classes = self.model.model.names
        self.reverse_classes = {v: k for k, v in self.classes.items()}
        self.calibrate()
        self.nc = len(self.classes)

    def class_id_to_name(self, c):
        return self.classes[c]

    def class_name_to_idx(self, c, raise_error=False):
        try:
            return self.reverse_classes[c]
        except KeyError as e:
            if raise_error:
                raise e

            return 123456789

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, *args, **kwargs):
        train_results = self.model.train(*args, **kwargs)
        best_path = train_results.save_dir / "weights/best.pt"
        return train_results, type(self)(best_path)

    def export(
        self,
        # See the following for a list of supported formats
        # https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/engine/exporter.py
        format: str,
    ) -> Path:
        return Path(self.model.export(format=format, verbose=False))

    def get_custom_tf_signature(
        self,
        default_signature: ConcreteFunction,
        dino_model: Path | str | None = None,
        score_thresholds: None = None,
    ) -> ConcreteFunction:
        """Get a custom tensorflow function signature with pre/post-processing.

        The custom signature adds preprocessing and postprocessing steps and
        optionally also returns an embedding for each detected object using the
        provided DINOv2 model.

        Args:
            default_signature: The default tensorflow signature for YOLOv8
            dino_model: A path to a DINOv2 model in TF SavedModel format.
                If provided, the output of the `serving_predictions` signature
                will include an embedding for each detected object.
        """
        if score_thresholds is not None:
            raise NotImplementedError(
                "score_thresholds is not implemented for single-label YOLO."
            )
        labels = [self.classes[i] for i in range(len(self.classes))]
        return get_custom_tf_signature(
            default_signature, labels, dino_model=dino_model
        )

    def export_tf_with_custom_signature(
        self,
        output_directory: Path | str | None = None,
        dino_model: Path | str | None = None,
    ) -> Path:
        """Export in the tensorflow SavedModel format with a custom signature.

        The custom signature adds preprocessing and postprocessing steps and
        optionally also returns an embedding for each detected object using the
        provided DINOv2 model.

        Args:
            output_directory: Where to write the model to.
                If None it will be written to the same directory as the input.
            dino_model: A path to a DINOv2 model in TF SavedModel format.
                If provided, the output of the `serving_predictions` signature
                will include an embedding for each detected object.
        """
        print("Exporting model as tf saved model.")
        tf_path = self.export("saved_model")

        if output_directory is None:
            output_directory = tf_path.parent

        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)

        tf_path_custom = output_directory / f"{tf_path.stem}_custom"

        # Enhance the tf model and save as a new tf model
        print("Adding custom signatures to tf model.")
        tf_model = tf.saved_model.load(tf_path)

        default_signature = tf_model
        prediction_signature = self.get_custom_tf_signature(
            default_signature, dino_model, self.calibration
        )

        # Adjust serving default to be better usage
        tf.saved_model.save(
            tf_model,
            tf_path_custom,
            signatures={
                "serving_default": default_signature,
                "serving_predictions": prediction_signature,
            },
        )

        return tf_path_custom

    def validate(
        self,
        df: pd.DataFrame | None = None,
        split: Literal["val", "train"] | None = "val",
        iou_threshold=0.5,
        score_threshold=0.25,
        **kwargs,
    ) -> "YOLOValidation":
        """Run validation the model and return a YOLOValidation object.

        See `YOLOValidation` for more info about the arguments and the
        functionality of teh YOLOValidation class.
        """
        val = None
        if split:
            val = self.model.val(
                save_json=True,
                iou=iou_threshold,
                conf=score_threshold,
                split=split,
            )

        return YOLOValidation(
            self,
            df=df,
            val=val,
            split=split,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            **kwargs,
        )

    def calibrate(self, calibration: dict | None = None, overwrite=False):
        """Load or save a calibration file for this model.

        The calibration file provides per-class confidence thresholds. See
        YOLOValidation.get_score_threshold_dict for details on the calibration
        process.

        It is stored in the same directory and name as the model with the
        suffix `-calibration.json`.

        Args:
            calibration: A dict describing the thresholds. If None, try to load
                the calibration from the path described above.
            overwrite: Whether, if calibration is provided, to overwrite the
                calibration json file with the newly passed calibration dict.

        Raises:
            FileExistsError: Raised if calibration is provided, the file
                exists and overwrite=False.
        """
        self.calibration_path = (
            self.path.parent / f"{self.path.stem}-calibration.json"
        )

        if calibration:
            if not self.calibration_path.exists() or overwrite:
                with self.calibration_path.open("w") as f:
                    json.dump(calibration, f)
            else:
                raise FileExistsError(
                    f"{self.calibration_path} exists. Set overwrite=True to overwrite."
                )

        self.calibration = None
        if self.calibration_path.exists():
            with self.calibration_path.open() as f:
                self.calibration = json.load(f)

    def generate_predictions(self, df, batch_size=32):
        """Generate predictions for the given df.

        Only returns positive detections. Doesn't return any rows for images
        where no detiction was found.

        Args:
            df: A DataFrame with an image_id and path/image_source column.
            batch_size: The number of images to predict at a time.

        Returns:
            A DataFrame with the following columns:
                image_id, path,
                image_width, image_height,
                pred, score,
                x1, y1, x2, y2
        """
        # Import here to avoid circular import
        from seegull.data.image import get_image_df,load_images_from_df 
        if "image_source" in df.columns
            image_df = get_image_df(df)
        

        paths = image_df[image_df["exists"]]["path"].tolist()
        dfs = []

        for paths_chunk in tqdm(
            np.array_split(paths, int(np.ceil(len(paths) / batch_size))),
            desc="Predicting",
        ):
            results = self.model(paths_chunk.tolist(), verbose=False)
            for result in results:
                d = pd.DataFrame(
                    np.hstack(
                        [
                            result.boxes.cls.cpu().unsqueeze(1),
                            result.boxes.xyxy.cpu(),
                            result.boxes.conf.cpu().unsqueeze(1),
                        ]
                    ),
                    columns=["target_id", "x1", "y1", "x2", "y2", "score"],
                )
                d["path"] = Path(result.path)
                d["image_height"], d["image_width"] = result.orig_shape
                dfs.append(d)

        predictions_df = pd.concat(dfs).reset_index(drop=True)
        predictions_df["pred"] = predictions_df["target_id"].apply(
            lambda i: self.classes[i]
        )

        return image_df[["image_id", "path"]].merge(predictions_df, on="path")[
            [
                "image_id",
                "image_width",
                "image_height",
                "pred",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        ]


class YOLOMultiLabel(YOLO):
    """A wrapper class for ultralytics_bower.YOLO

    ultralytics_bower is a fork of ultralytics which enables YOLOv8 models
    to predict two labels per object detection.
    """

    def __init__(self, model_path: str, transfer_weights: str | None = None):
        """
        Args:
            model_path: The path to the ultralytics_bower.YOLO model. Can be a
                .pt file for a trained model or a .yaml file for the base
                configuration.
            transfer_weights: An optional path to an ultralytics.YOLO model
                (NOTE: This is the default YOLOv8 implementation, not the
                custom one created in ultralytics_bower). This will transfer
                the weights for all of the common layers from a pretrained
                YOLOv8 model to the YOLOMultiLabel model.
        """
        import ultralytics_bower

        ultralytics_bower.checks()
        self.ultralytics = "ultralytics_bower"
        self.path = Path(model_path)
        self.model = ultralytics_bower.YOLO(model_path)
        self.calibrate()

        if transfer_weights:
            m = ultralytics_bower.YOLO(transfer_weights)
            self.model.load_state_dict(m.model.state_dict(), strict=False)

        names = self.model.model.names
        if isinstance(names, list):
            names = {i: v for i, v in enumerate(names)}

        self.classes = {
            tuple(p): tuple([names[i][j] for i, j in enumerate(p)])
            for p in itertools.product(*names.values())
        }
        self.reverse_classes = {v: k for k, v in self.classes.items()}

        self.tuple_to_idx = {t: i for i, t in enumerate(self.classes.keys())}
        self.idx_to_name = {
            i: self.classes[t] for t, i in self.tuple_to_idx.items()
        }

        # Get the number of classes for each label
        self.nc = tuple(np.array(list(self.classes.keys())).max(0) + 1)

    def class_id_to_name(self, c):
        return self.classes[tuple(c)]

    def class_name_to_idx(self, c, raise_error=False):
        try:
            return self.tuple_to_idx[self.reverse_classes[c]]
        except KeyError as e:
            if raise_error:
                raise e

            return 123456789

    def generate_predictions(self, df: pd.DataFrame, batch_size: int = 32):
        """Generate predictions for the given df.

        Only returns positive detections. Doesn't return any rows for images
        where no detiction was found.

        Args:
            df: A DataFrame with an image_id and path/image_source column.
            batch_size: The number of images to predict at a time.

        Returns:
            A DataFrame with the following columns:
                image_id, path,
                image_width, image_height,
                pred, score,
                x1, y1, x2, y2
        """
        # Import here to avoid circular import
        from seegull.data.image import get_image_df

        image_df = get_image_df(df)

        paths = image_df[image_df["exists"]]["path"].tolist()
        dfs = []

        for paths_chunk in tqdm(
            np.array_split(paths, int(np.ceil(len(paths) / batch_size))),
            desc="Predicting",
        ):
            results = self.model(paths_chunk.tolist(), verbose=False)
            for result in results:
                d = pd.DataFrame(
                    np.hstack(
                        [
                            result.boxes.cls.cpu(),
                            result.boxes.xyxy.cpu(),
                            result.boxes.conf.cpu(),
                        ]
                    ),
                    columns=[
                        "object_id",
                        "material_id",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "score0",
                        "score1",
                    ],
                )
                d["path"] = Path(result.path)
                d["image_height"], d["image_width"] = result.orig_shape
                dfs.append(d)

        predictions_df = pd.concat(dfs).reset_index(drop=True)
        predictions_df["object"] = predictions_df["object_id"].apply(
            lambda i: self.model.model.names[0][i]
        )
        predictions_df["material"] = predictions_df["material_id"].apply(
            lambda i: self.model.model.names[1][i]
        )
        predictions_df["pred"] = predictions_df[["object", "material"]].apply(
            tuple, axis=1
        )
        predictions_df["score"] = predictions_df[["score0", "score1"]].apply(
            list, axis=1
        )

        return image_df[["image_id", "path"]].merge(predictions_df, on="path")[
            [
                "image_id",
                "image_width",
                "image_height",
                "score",
                "pred",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        ]

    def get_custom_tf_signature(
        self,
        default_signature: ConcreteFunction,
        dino_model: Path | str | None = None,
        score_thresholds: dict | None = None,
    ) -> tuple[ConcreteFunction, ConcreteFunction]:
        """Get a custom tensorflow function signature with pre/post-processing.

        The custom signature adds preprocessing and postprocessing steps and
        optionally also returns an embedding for each detected object using the
        provided DINOv2 model.

        Args:
            default_signature: The default tensorflow signature for YOLOv8
            dino_model: A path to a DINOv2 model in TF SavedModel format.
                If provided, the output of the `serving_predictions` signature
                will include an embedding for each detected object.
            score_thresholds: A dict containing pre-class thresholds for
                considering a given score a positive detection.
        """
        # Construct a 2d array mapping class combinations to labels
        nc = self.nc
        labels = np.array(
            [[""] * (nc[1] + 1) for _ in range(nc[0] + 1)], dtype=object
        )
        # Add "unknown" combinations at the last index of each axis of the array
        for k, v in self.classes.items():
            labels[k[0], k[1]] = "_".join(v)
            labels[k[0], nc[1]] = "_".join([v[0], "unknown"])
            labels[nc[0], k[1]] = "_".join(["unknown", v[1]])

        # Load the confidence threshold calibration (if provided) from a dict
        if score_thresholds:
            score_threshold_arrays = []

            for i, k in enumerate(["object", "material"]):
                k_confs = []

                names = self.model.names[i]
                for j in range(len(names)):
                    k_confs.append(score_thresholds[k].get(names[j], 1))

                score_threshold_arrays.append(np.array(k_confs))
        else:
            score_threshold_arrays = [np.zeros(i) for i in nc]

        score_thresholds = np.concatenate(score_threshold_arrays)

        return get_custom_tf_signature(
            default_signature,
            labels,
            nc=nc,
            multilabel=True,
            dino_model=dino_model,
            score_thresholds=score_thresholds,
        )


class YOLOValidation:
    """A class which provides assorted validation functions."""

    def __init__(
        self,
        model: YOLO,
        df: pd.DataFrame | None = None,
        val: "ultralytics.utils.metrics.DetMetrics" = None,
        split: Literal["val", "train"] | None = None,
        predictions_csv: str | None = None,
        iou_threshold=0.5,
        score_threshold=0.25,
        target: str | tuple[str] = "target",
    ):
        """
        Accepts predictions and ground truth in several different formats:
            - Validation results from ultralytics' default validation function.
                This provides both the ground truth and predictions.
            - A CSV in a specified format that provides predictions (see
                `load_predictions_csv` for details on format).
            - A DataFrame of images with ground truth annotations.
                Predictions will be generated for those images if predictions
                aren't provided using one of the above methods.

        Args:
            model: The YOLO model to validate. Either YOLO or YOLOMulitLabel.
            df:  A DataFrame of images with ground truth annotations.
            val: A validation object produced by ultralytics.YOLO.val().
            split: If `val` is provided, the split used for creating it.
            predictions_csv: A CSV of predictions for images provided in df.
            iou_threshold: The minimum intersection-over-union to consider
                an object detection the same when comparing predictions
                with ground truth.
            score_threshold: The minimum score/confidence to inclue a
                prediction.
            target: The target to validate (such as "object" or "material").
                Either the model's target or in the case of YOLOMultiLabel
                it's also possible to pass a subset of the model's target.

        Raises:
            ValueError: If none of val, predictions_csv or df is provided

        """
        from seegull.data.image import BoxAndLabelAnnotator

        self.model = model
        self.df = df
        # Store the default values passed in case of calibration since
        # calibration will change iou_threshold and score_threshold
        self._default_iou_threshold = self.iou_threshold = iou_threshold
        self._default_score_threshold = self.score_threshold = score_threshold

        # Turn target into a list if it's a string for consistency
        self.target = [target] if isinstance(target, str) else target

        # Load the predictions using one of the provided inputs
        self._predictions_df = None
        self.val = val
        if val:
            # Load the image ids that were included in this split
            # so we can subset df to only include those.
            ultralytics = __import__(model.ultralytics)
            dataset = ultralytics.data.utils.check_det_dataset(
                model.model.overrides["data"]
            )
            self.image_ids = [p.stem for p in Path(dataset[split]).glob("*")]
        elif predictions_csv:
            self.predictions_csv = Path(predictions_csv)
            self._predictions_df = self.load_predictions_csv(predictions_csv)
        elif df is not None:
            self._predictions_df = self.generate_predictions(self.df)
        else:
            raise ValueError(
                "One of `val`, `predictions_csv` or `df` must be provided to YOLOValidation."
            )

        self.box_annotator = BoxAndLabelAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.CENTER
        )

    def load_predictions_df(self):
        """Load the predictions from self.val and return them as a DataFrame."""
        if self._predictions_df is not None:
            return self._predictions_df

        # Load the predictions from running validation
        with (self.val.save_dir / "predictions.json").open("r") as f:
            predictions = json.load(f)

        predictions_df = pd.DataFrame(predictions)
        predictions_df["image_id"] = predictions_df["image_id"].astype(str)

        # Convert the class IDs to names
        predictions_df["pred"] = predictions_df["category_id"].apply(
            lambda c: self.model.class_id_to_name(c)
        )

        # Unpack the bounding boxes in xywh format to xyxy
        predictions_df[["x1", "y1", "x2", "y2"]] = predictions_df.apply(
            lambda row: (
                row.bbox[0],
                row.bbox[1],
                row.bbox[0] + row.bbox[2],
                row.bbox[1] + row.bbox[3],
            ),
            axis=1,
            result_type="expand",
        )

        # TODO: Load the dimensions dynamically from the model's parameters
        predictions_df["image_width"], predictions_df["image_height"] = 640, 640

        self._predictions_df = predictions_df
        return predictions_df

    def generate_predictions(self, df, *args, **kwargs):
        """Generate predictions for the given DataFrame of images.

        Return them in the same format as `load_predictions_df`.

        Args:
            df: A DataFrame with an image_id and path (or image_source) column.
        """
        predictions_df = self.model.generate_predictions(df, *args, **kwargs)

        # Store the image ids so we know which images the validation is run on.
        self.image_ids = df["image_id"].unique().tolist()

        return predictions_df

    def load_predictions_csv(self, csv_path):
        """Load predictions from a CSV.

        The CSV requires the following columns:
            - image_id
            - image_width
            - image_height
            - x1
            - y1
            - x2
            - y2
            - *Columns corresponding with the given target(s)*
                - object/object_type
                - object_confidence
                - material/material_type
                - material_confidence
                - brand
                - brand_confidence
        """
        df = pd.read_csv(csv_path, on_bad_lines="skip").rename(
            columns={"object_type": "object", "material_type": "material"}
        )
        df["image_id"] = df["image_id"].astype(str)

        # Save image ids that the test was run on
        self.image_ids = df["image_id"].unique().tolist()

        # Remove any where there was an error. These might be failed detections(?)
        df = df[~(df == "error").any(axis=1)]

        if len(self.target) > 1:
            df["score"] = df[
                ["object_confidence", "material_confidence"]
            ].apply(list, axis=1)
            df["pred"] = df[["object", "material"]].apply(tuple, axis=1)
        else:
            df["score"] = df[f"{self.target[0]}_confidence"]
            df["pred"] = df[self.target[0]]

        return df[
            [
                "image_id",
                "image_width",
                "image_height",
                "score",
                "pred",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        ]

    def subset_df(self, df, image_ids=None):
        """Return a subset of the given df with only the specified images."""
        if image_ids is None:
            image_ids = self.image_ids

        return df[df["image_id"].isin(image_ids)].reset_index(drop=True)

    def _plot_confusion_matrix(
        self, confusion_matrix, labels, figsize, title=""
    ):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=labels
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        disp.plot(ax=ax, xticks_rotation="vertical")

    def _merged_df(self, image_ids=None, include_image_cols=False):
        """Merge the predictions with the ground truth df."""
        # Load the predictions
        predictions_df = self.subset_df(self.load_predictions_df(), image_ids)

        if len(self.target) > 1:
            # Remove predictions with score less than score threshold on
            # either of the classes
            predictions_df = predictions_df[
                (predictions_df["score"].str[0] >= self.score_threshold)
                & (predictions_df["score"].str[1] >= self.score_threshold)
            ].reset_index(drop=True)

            for i, col in enumerate(self.target):
                predictions_df[f"pred_{col}"] = predictions_df["pred"].apply(
                    lambda v: v[i]
                )
        else:
            # Remove predictions with score less than score threshold
            predictions_df = predictions_df[
                (predictions_df["score"] >= self.score_threshold)
            ].reset_index(drop=True)
            predictions_df[f"pred_{self.target[0]}"] = predictions_df["pred"]

        val_df = self.subset_df(self.df, image_ids)

        # Set up the labels we'll use
        if len(self.target) > 1:
            val_df["true"] = val_df[self.target].apply(list, axis=1)
            val_df["true_str"] = val_df["true"].apply(
                lambda v: "+".join(v) if not pd.isnull(v).any() else None
            )
            predictions_df["pred_str"] = predictions_df["pred"].apply(
                lambda v: None
                if (pd.isnull(v) or pd.isnull(list(v)).any())
                else "+".join(v)
            )
        else:
            val_df["true"] = val_df[self.target[0]]
            val_df["true_str"] = val_df["true"].astype(str)
            predictions_df["pred_str"] = predictions_df["pred"].astype(str)

        # Create a unique ID on both dfs before merging to track which
        # rows were dropped.
        predictions_df["id"] = np.arange(len(predictions_df))
        val_df["id"] = np.arange(len(val_df))

        # Merge the two on image_id
        val_cols = [
            "id",
            "image_id",  #'path',
            "image_width",
            "image_height",
            "x1",
            "y1",
            "x2",
            "y2",
            "true",
            "true_str",
        ] + self.target
        merged_df = predictions_df.merge(
            val_df[val_cols], how="inner", on="image_id", suffixes=["_p", "_g"]
        )

        # Calculate the iou between the ground truth and prediction
        w, h = merged_df["image_width_p"], merged_df["image_height_p"]
        sw, sh = merged_df["image_width_g"] / w, merged_df["image_height_g"] / h
        scale = np.dstack([sw, sh, sw, sh]).squeeze()
        merged_df["iou"] = iou(
            merged_df[["x1_g", "y1_g", "x2_g", "y2_g"]].values / scale,
            merged_df[["x1_p", "y1_p", "x2_p", "y2_p"]].values,
        )
        merged_df["iou_gte_threshold"] = merged_df["iou"] >= self.iou_threshold

        # Subset the merged_df to >= iou threshould
        merged_df = merged_df[merged_df["iou_gte_threshold"]]

        # Bring together the matches with the missing rows as "background"
        shared_cols = [
            "image_width",
            "image_height",
            "x1",
            "y1",
            "x2",
            "y2",
            "id",
        ]
        merged_df = (
            pd.concat(
                [
                    merged_df,
                    predictions_df[
                        ~predictions_df["id"].isin(merged_df["id_p"])
                    ].rename(columns={col: f"{col}_p" for col in shared_cols}),
                    val_df[~val_df["id"].isin(merged_df["id_g"])][
                        val_cols
                    ].rename(columns={col: f"{col}_g" for col in shared_cols}),
                ]
            )
            .reset_index(drop=True)
            .fillna(
                {
                    "true_str": "background",
                    "pred_str": "background",
                    "iou": 0,
                    "iou_gte_threshold": False,
                }
                | {f"pred_{t}": "background" for t in self.target}
                | {t: "background" for t in self.target}
            )
        )

        merged_df = (
            val_df[["image_id", "path"]]
            .drop_duplicates()
            .merge(merged_df)
            .reset_index(drop=True)
        )

        return merged_df

    def metrics_per_class(self, target=None, subset_df_func=None):
        if target is None:
            target = self.target

        merged_df = self._merged_df()

        if subset_df_func:
            merged_df = subset_df_func(merged_df)

        if isinstance(target, str):
            col = target
            true = merged_df[target]
            pred = merged_df[f"pred_{target}"]
        else:
            col = "index"
            true = merged_df[target].agg("+".join, axis=1)
            pred = merged_df[[f"pred_{c}" for c in target]].agg(
                "+".join, axis=1
            )

        labels = sorted(set(true.unique()))

        df = pd.DataFrame(
            {
                col: labels,
                "precision": precision_score(
                    true, pred, average=None, labels=labels
                ),
                "recall": recall_score(true, pred, average=None, labels=labels),
                "f1": f1_score(true, pred, average=None, labels=labels),
            }
        ).merge(true.value_counts().reset_index())

        if not isinstance(target, str):
            df[target] = df["index"].str.split("+", expand=True)
            df = df[target + ["precision", "recall", "f1", "count"]]

        df = df.replace("background", None).dropna()

        count = df["count"].sum()
        mean = pd.DataFrame(df[["precision", "recall", "f1"]].mean()).T
        weighted_mean = pd.DataFrame(
            {
                "precision": [(df["precision"] * df["count"]).sum() / count],
                "recall": [(df["recall"] * df["count"]).sum() / count],
                "f1": [(df["f1"] * df["count"]).sum() / count],
            }
        )

        df = pd.concat([df, mean]).fillna({"count": count}).fillna("MEAN")

        df = (
            pd.concat([df, weighted_mean])
            .fillna({"count": count})
            .fillna("WEIGHTED MEAN")
        )

        return df

    def _multilabel_confusion_matrix(self, figsize):
        # Set up the dataframe with the correct ground truth and prediction
        # labels
        merged_df = self._merged_df()

        # Get the sorted list of labels with "background" last
        labels = sorted(
            (
                set(merged_df["true_str"].unique())
                | set(merged_df["pred_str"].unique())
            )
            - {"background"}
        ) + ["background"]

        # Compute and plot the confusion matrix
        cm = confusion_matrix(
            merged_df["true_str"], merged_df["pred_str"], labels=labels
        )
        self._plot_confusion_matrix(cm, labels, figsize, str(self.target))

        # Repeat the above process for each target individually
        for col in self.target:
            labels = sorted(
                (
                    set(merged_df[col].unique())
                    | set(merged_df[f"pred_{col}"].unique())
                )
                - {"background"}
            ) + ["background"]
            cm = confusion_matrix(
                merged_df[col], merged_df[f"pred_{col}"], labels=labels
            )

            self._plot_confusion_matrix(cm, labels, figsize, col)

    def confusion_matrix(self, figsize=(20, 20)):
        if len(self.target) == 1:
            raise NotImplementedError(
                "single target confusion_matrix is not implemented for manual validation"
            )
            self._plot_confusion_matrix(
                self.val.confusion_matrix.matrix.astype(int),
                list(self.val.names.values()) + ["background"],
                figsize,
            )
        else:
            self._multilabel_confusion_matrix(figsize)

    def print_metrics(self):
        if not self.val:
            raise NotImplementedError(
                "print_metrics is not implemented for manual validation"
            )
        for n, v in zip(self.val.keys, self.val.mean_results()):
            print(f"{n}: {v}")

    def display_discrepencies(
        self,
        resize=0.5,
        padding=30,
        limit: int = None,
        image_ids=None,
    ):
        """Display discrepencies between the ground truth and predictions.

        Images are displayed with box annotations for the predictions and
        label annotations in the center of each ground truth object.

        Only discrepencies are displayed. If a box and label are both shown
        on an object, then the class is mismatched. If only a box is shown
        without a label, then there is no corresponding ground truth
        for that prediction. If a label is shown without a box, then there
        was no prediction for that ground truth object.

        Args:
            resize: A float with a percentage to resize the image by. This is
                useful for large images so the annotation text is legible.
            padding: How much padding to put around each image. Helps if the
                bounding box is near the edge of an image.
            limit: The maximum number of descrepencies to display
            image_ids: A list of image ids. Only show images in this subset.
                An image in this subset without any discrepencies won't be
                shown.
        """
        # Import here to avoid circular import
        from seegull.data.image import load_image

        merged_df = self._merged_df(
            image_ids=image_ids, include_image_cols=False
        )

        if len(self.target) > 1:
            merged_df["true"] = merged_df["true"].apply(
                lambda t: tuple(t) if isinstance(t, list) else None
            )

        discrepencies_df = merged_df[
            merged_df["true_str"] != merged_df["pred_str"]
        ]

        if limit:
            discrepencies_df = discrepencies_df[
                discrepencies_df["image_id"].isin(
                    discrepencies_df["image_id"].drop_duplicates().head(limit)
                )
            ]

        def plot_merged_df_image(rows):
            im = load_image(rows.iloc[0], resize=resize)
            image_height_g, image_width_g = rows.iloc[0][
                ["image_height_g", "image_width_g"]
            ]
            image_height_p, image_width_p = rows.iloc[0][
                ["image_height_p", "image_width_p"]
            ]

            def format_prediction_label(row):
                if isinstance(row.score, list):
                    score = ", ".join(f"{v:.2f}" for v in row.score)
                else:
                    score = f"{row.score:.2f}"

                return f"Prediction: {row.pred_str} ({score})"

            def create_data_dict(w, h, label, col, suffix):
                nonbackground = rows[rows[f"{col}_str"] != "background"]
                if len(nonbackground) == 0:
                    return {
                        "xyxy": np.array([]),
                        "classes": np.array([]),
                        "labels": np.array([]),
                    }

                h, w = nonbackground.iloc[0][
                    [f"image_height{suffix}", f"image_width{suffix}"]
                ]
                sw, sh = im.width / w, im.height / h
                scale = np.array([sw, sh, sw, sh])

                return {
                    "xyxy": nonbackground[
                        [
                            f"x1{suffix}",
                            f"y1{suffix}",
                            f"x2{suffix}",
                            f"y2{suffix}",
                        ]
                    ].values
                    * scale,
                    "classes": nonbackground[col].apply(
                        self.model.class_name_to_idx
                    ),
                    "labels": [f"{label}: {c}" for c in nonbackground[col]]
                    if col == "true"
                    else [
                        format_prediction_label(row)
                        for _, row in nonbackground.iterrows()
                    ],
                }

            data = {
                i: create_data_dict(w, h, label, col, suffix)
                for i, w, h, label, col, suffix in [
                    (
                        "ground_truth",
                        image_width_g,
                        image_height_g,
                        "Ground Truth",
                        "true",
                        "_g",
                    ),
                    (
                        "predictions",
                        image_width_p,
                        image_height_p,
                        "Prediction",
                        "pred",
                        "_p",
                    ),
                ]
            }

            print(rows["image_id"].iloc[0])
            self.plot_image_with_data(im, data, padding=padding)

        discrepencies_df.groupby("image_id").apply(plot_merged_df_image)

    def plot_image_with_data(
        self, im, data, padding=30, annotators=None, size=(12, 12)
    ):
        if annotators is None:
            annotators = [self.label_annotator, self.box_annotator]

        frame = cv2.copyMakeBorder(
            im.image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        for t, annotator in zip(["ground_truth", "predictions"], annotators):
            if len(data[t]["xyxy"]):
                frame = annotator.annotate(
                    scene=frame,
                    detections=sv.Detections(
                        xyxy=np.array(data[t]["xyxy"]) + padding,
                        class_id=np.array(data[t]["classes"]),
                    ),
                    labels=data[t]["labels"],
                )

        sv.plot_image(frame, size=size)

    def plot_image_by_id(self, image_id, resize=0.5, **kwargs):
        # Import here to avoid circular import
        from seegull.data.image import load_image

        ground_truth = self.df[self.df["image_id"] == image_id].copy()

        if len(self.target) == 1:
            ground_truth["target"] = ground_truth[self.target]
        else:
            ground_truth["target"] = ground_truth[self.target].apply(
                tuple, axis=1
            )

        im = load_image(ground_truth.iloc[0], resize=resize)
        image_height, image_width = ground_truth.iloc[0][
            ["image_height", "image_width"]
        ]

        predictions_df = self.load_predictions_df()
        predictions = predictions_df[
            predictions_df["image_id"] == image_id
        ].copy()
        if len(predictions):
            p_image_height, p_image_width = predictions.iloc[0][
                ["image_height", "image_width"]
            ]
        else:
            p_image_height, p_image_width = image_height, image_width

        def format_prediction_label(row):
            if isinstance(row.score, list):
                score = ", ".join(f"{v:.2f}" for v in row.score)
            else:
                score = f"{row.score:.2f}"

            return f"Prediction: {row.pred} ({score})"

        def create_data_dict(df, w, h, label, col):
            sw, sh = im.width / w, im.height / h
            scale = np.array([sw, sh, sw, sh])

            return {
                "xyxy": df[["x1", "y1", "x2", "y2"]].values * scale,
                "classes": df[col].apply(self.model.class_name_to_idx),
                "labels": [f"{label}: {c}" for c in df[col]]
                if col == "target"
                else [format_prediction_label(row) for _, row in df.iterrows()],
            }

        data = {
            i: create_data_dict(df, w, h, label, col)
            for i, df, w, h, label, col in [
                (
                    "ground_truth",
                    ground_truth,
                    image_width,
                    image_height,
                    "Ground Truth",
                    "target",
                ),
                (
                    "predictions",
                    predictions,
                    p_image_width,
                    p_image_height,
                    "Prediction",
                    "pred",
                ),
            ]
        }

        self.plot_image_with_data(im, data, **kwargs)

    # TODO: Maybe move this to a separate class that is specific to
    # material+object since the rest of this is generic?
    def threshold_comparison(
        self,
        threshold: Literal["iou", "score"],
        target=None,
        step=0.05,
    ):
        if target is None:
            target = self.target

        steps = np.arange(step, 1 + step, step)
        dfs = []

        for s in steps:
            setattr(self, f"{threshold}_threshold", s)
            metrics = self.metrics_per_class(target)
            metrics[threshold] = s
            dfs.append(metrics)

        df = pd.concat(dfs)
        best_threshold_by_class = (
            df.sort_values("f1", ascending=False).groupby(target).head(1)
        )

        return best_threshold_by_class.sort_values("count", ascending=False)

    def get_score_threshold_dict(
        self,
        f1_threshold=0.25,
        step=0.05,
    ):
        # Calibrate iou before calibrating score
        self.score_threshold = self._default_score_threshold
        iou_comp_df = self.threshold_comparison("iou", step=step)
        self.iou_threshold = iou_comp_df[iou_comp_df["f1"] > 0]["iou"].mean()

        material_df = self.threshold_comparison(
            "score", step=step, target="material"
        )
        object_df = self.threshold_comparison(
            "score", step=step, target="object"
        )

        return (
            {
                "object": {
                    row.object: row.score if row.f1 > f1_threshold else 1
                    for _, row in object_df.iterrows()
                    if "MEAN" not in row.object
                },
                "material": {
                    row.material: row.score if row.f1 > f1_threshold else 1
                    for _, row in material_df.iterrows()
                    if "MEAN" not in row.material
                },
            },
            object_df,
            material_df,
        )

    def calibrate_model(self, *args, overwrite=False, **kwargs):
        (calibration_dict, object_df, material_df) = (
            self.get_score_threshold_dict(*args, **kwargs)
        )
        self.model.calibrate(calibration_dict, overwrite=overwrite)

        return calibration_dict, object_df, material_df


def get_custom_tf_signature(
    default_signature: ConcreteFunction,
    labels: list[str] | np.ndarray,
    multilabel=False,
    dino_model: Path | str | None = None,
    score_thresholds: np.ndarray | None = None,
    nc: int | list[int] | None = None,
) -> ConcreteFunction:
    if dino_model:
        dino_signature = tf.saved_model.load(dino_model).signatures[
            "with_preprocessing"
        ]
    else:
        dino_signature = None

    @tf.function
    def get_detections(
        image_raw,
        model_shape=(640, 640),
        image_shape=(640, 640),
    ):
        # Load image into (image_height, image_width, 3)
        image_bytes = tf.reshape(
            tf.io.decode_raw(image_raw, tf.uint8),
            [image_shape[0], image_shape[1], 3],
        )

        # Calculate the scaling factor between the model and image shapes
        scale = tf.reverse(tf.cast(image_shape, tf.float32) / model_shape, [0])

        # Resize to the model shape if the image isn't already the correct size
        if (model_shape[0] != image_shape[0]) or (
            model_shape[1] != image_shape[1]
        ):
            resized_image = tf.cast(
                tf.image.resize(image_bytes, model_shape), tf.uint8
            )
        else:
            resized_image = image_bytes

        normalized_image = tf.expand_dims(
            tf.cast(resized_image, dtype=tf.float32) / 255, 0
        )

        detections = tf.cast(
            tf.transpose(default_signature(normalized_image)[0]),
            dtype=tf.float32,
        )

        # Convert cx,cy,w,h to x1y1x2y2
        xy = detections[:, :2]
        wh = detections[:, 2:4] / 2.0
        scores = detections[:, 4:]

        # Mask the scores by whether they're greater than their class threshold
        if score_thresholds is not None:
            scores = scores * tf.cast(scores > score_thresholds, tf.float32)

        detections = tf.concat(
            [
                tf.concat(
                    [
                        (xy - wh) * scale,
                        (xy + wh) * scale,
                    ],
                    1,
                ),
                scores,
            ],
            1,
        )

        return image_bytes, detections

    @tf.function
    def dinov2(
        args,
    ):
        image_bytes, bbox = args

        # Crop the image to the bounding box
        bounding_box_image = image_bytes[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        # Convert BGR->RGB
        bounding_box_image = tf.reverse(bounding_box_image, axis=[2])

        return dino_signature(image=bounding_box_image)["embedding"]

    if not multilabel:

        @tf.function
        def get_predictions(
            image_raw,
            model_shape=(640, 640),
            image_shape=(640, 640),
            score_threshold=0.80,
            iou_threshold=0.50,
            max_boxes=20,
            return_embeddings=False,
        ):
            # Perform the preprocessing and get the raw output from the model
            image_bytes, detections = get_detections(
                image_raw, model_shape, image_shape
            )

            # Get max score for each row
            scores = tf.math.reduce_max(detections[:, 4:], axis=1)

            # Apply NMS and filter the detections
            applied_indices = tf.image.non_max_suppression(
                detections[:, :4],
                scores,
                max_boxes,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            detections = tf.gather(detections, applied_indices)

            # Get the scores and classes for each detection
            conf = tf.math.reduce_max(detections[:, 4:], axis=1)
            classes = tf.math.argmax(detections[:, 4:], axis=1)
            classes = tf.gather(labels, classes)
            boxes = detections[:, :4]

            response = {
                "xyxy": boxes,
                "classes": classes,
                "conf": conf,
            }

            if dino_signature and return_embeddings:
                response["dinov2_embeddings"] = tf.map_fn(
                    dinov2,
                    [
                        tf.tile(
                            tf.expand_dims(image_bytes, 0),
                            [tf.shape(boxes)[0], 1, 1, 1],
                        ),
                        tf.cast(boxes, tf.int32),
                    ],
                    fn_output_signature=tf.TensorSpec(
                        [dino_signature.output_shapes["embedding"][-1]],
                        tf.float32,
                    ),
                )
            else:
                response["dinov2_embeddings"] = tf.zeros(0)

            return response

    else:
        # Get the number of classes for each of the labels
        if nc is None:
            nc = labels.shape

        @tf.function
        def get_predictions(
            image_raw,
            model_shape=(640, 640),
            image_shape=(640, 640),
            score_threshold=0.80,
            iou_threshold=0.50,
            max_boxes=20,
            return_embeddings=True,
        ):
            # Perform the preprocessing and get the raw output from the model
            image_bytes, detections = get_detections(
                image_raw, model_shape, image_shape
            )

            # Get the  score for each row by multiplying the individual confidences
            scores = tf.math.reduce_max(
                [
                    tf.math.reduce_max(v, axis=1)
                    for v in tf.split(detections, [4] + list(nc), axis=1)[1:]
                ],
                axis=0,
            )

            # Apply NMS and filter the detections
            applied_indices = tf.image.non_max_suppression(
                detections[:, :4],
                scores,
                max_boxes,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            detections = tf.gather(detections, applied_indices)

            # Get the scores for each detection by either taking the product
            # if both are non-zero or taking the non-zero element if one
            # is zero.
            conf_by_target = [
                tf.math.reduce_max(v, axis=1)
                for v in tf.split(detections, [4] + list(nc), axis=1)[1:]
            ]
            conf_prod = tf.math.reduce_prod(conf_by_target, axis=0)
            conf_max = tf.math.reduce_max(conf_by_target, axis=0)
            conf = tf.where(conf_prod > 0, conf_prod, conf_max)

            # Convert to a text label
            classes = tf.stack(
                [
                    tf.math.argmax(
                        tf.pad(v, [[0, 0], [0, 1]], constant_values=1e-3),
                        axis=1,
                    )
                    for v in tf.split(detections, [4] + list(nc), axis=1)[1:]
                ],
                axis=1,
            )
            classes = tf.gather_nd(labels, classes)
            boxes = detections[:, :4]

            response = {
                "xyxy": boxes,
                "classes": classes,
                "conf": conf,
            }

            if dino_signature and return_embeddings:
                response["dinov2_embeddings"] = tf.map_fn(
                    dinov2,
                    [
                        tf.tile(
                            tf.expand_dims(image_bytes, 0),
                            [tf.shape(boxes)[0], 1, 1, 1],
                        ),
                        tf.cast(boxes, tf.int32),
                    ],
                    fn_output_signature=tf.TensorSpec(
                        [dino_signature.output_shapes["embedding"][-1]],
                        tf.float32,
                    ),
                )
            else:
                response["dinov2_embeddings"] = tf.zeros(0)

            return response

    # Create new signature, to get filtered boounding boxes
    prediction_signature = get_predictions.get_concrete_function(
        image_raw=tf.TensorSpec(
            tf.TensorShape([None]), dtype=tf.string, name="image_raw"
        ),
        image_shape=tf.TensorSpec(
            tf.TensorShape([2]), dtype=tf.int32, name="image_shape"
        ),
        score_threshold=tf.TensorSpec(
            shape=(), dtype=tf.float32, name="score_threshold"
        ),
        iou_threshold=tf.TensorSpec(
            shape=(), dtype=tf.float32, name="iou_threshold"
        ),
        max_boxes=tf.TensorSpec(shape=(), dtype=tf.int32, name="max_boxes"),
        return_embeddings=tf.TensorSpec(
            shape=(), dtype=tf.bool, name="return_embeddings"
        ),
    )

    return prediction_signature


def write_training_example(
    args,
    classes: list[str] | list[list[str]] = None,
    resize: tuple[int] = (640, 640),
    target: str | tuple[str] = "target",
):
    image_id, group = args

    # Save image
    image = group["image"].iloc[0].resize(*resize)
    image.save(group["image_path"].iloc[0])

    # Save annotations
    if len(group[target].dropna()):
        with group["annotation_path"].iloc[0].open("w") as f:
            for _, row in group.iterrows():
                cx = (row.x1 + row.x2) / 2 / row.image_width
                cy = (row.y1 + row.y2) / 2 / row.image_height
                w = (row.x2 - row.x1) / row.image_width
                h = (row.y2 - row.y1) / row.image_height

                # Single label
                if isinstance(target, str):
                    label = classes.index(row[target])
                else:
                    label = " ".join(
                        [
                            str(classes[i].index(row[target[i]]))
                            for i in range(len(target))
                        ]
                    )

                f.write(f"{label} {cx} {cy} {w} {h}\n")


def format_training_data(
    df: pd.DataFrame,
    output_path: Path | str,
    test_size=0.2,
    overwrite=False,
    target: str | tuple[str] = "target",
):
    """Create a directory with YOLO-format training data.

    If `target` is a single string then create standard yolo training data.
    If `target` is a tuple, create data in the custom multi-head format
    where each box has two labels.

    Requires df to have nine or ten columns:
    - image_id: The unique image id
    - image: A seegull.data.image.Image instance
        If image is not provided, but path or image_source is, load the images
    - image_width, image_height: The width and height of the image
    - x1, y1, x2, y2: Bounding box
    - target: One or two columns corresponding to target
    """
    from seegull.data.image import load_images

    output_path = Path(output_path)
    df = df.copy().reset_index()

    if "image" not in df.columns:
        df["image"] = load_images(df)

    print(df["image"])

    splits = {}
    for split in ["train", "test"]:
        splits[split] = (
            output_path / "images" / split,
            output_path / "labels" / split,
        )
        for path in splits[split]:
            if path.exists():
                if overwrite:
                    shutil.rmtree(path)
                    path.mkdir()
                else:
                    raise FileExistsError
            else:
                path.mkdir(parents=True)

    data_yaml = {
        "train": "images/train",
        "val": "images/test",
    }

    if isinstance(target, str):
        classes = sorted(df[target].dropna().unique())
        data_yaml["names"] = {i: n for i, n in enumerate(classes)}
        stratify_by = target
    else:
        classes = [sorted(df[t].unique()) for t in target]

        data_yaml["names"] = [{i: n for i, n in enumerate(c)} for c in classes]

        df["target_combined"] = df[target[0]] + "_" + df[target[1]]
        stratify_by = "target_combined"

    with (output_path / "data.yaml").open("w") as f:
        yaml.dump(data_yaml, f)

    unique_images = df.drop_duplicates(subset="image_id").reset_index(drop=True)
    train_ids, test_ids = train_test_split(
        unique_images["image_id"],
        test_size=test_size,
        stratify=unique_images[stratify_by].fillna("background"),
    )
    train_df = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_ids)].reset_index(drop=True)

    for split, subdf in [("train", train_df), ("test", test_df)]:
        subdf["image_path"] = subdf["image_id"].apply(
            lambda image_id: splits[split][0] / f"{image_id}.jpg"
        )
        subdf["annotation_path"] = subdf["image_id"].apply(
            lambda image_id: splits[split][1] / f"{image_id}.txt"
        )

        process_map(
            partial(write_training_example, classes=classes, target=target),
            list(subdf.groupby("image_id")),
            desc=f"writing {split} data",
        )
