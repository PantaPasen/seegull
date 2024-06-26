"""Contains wrappers for several different versions of DINO models:

- Huggingface Transformers' DINOv2:
    https://huggingface.co/docs/transformers/en/model_doc/dinov2
- A Tensorflow version of the official DINOv2 model:
    https://github.com/facebookresearch/dinov2/issues/19#issuecomment-1880139201
- GroundingDINO (https://github.com/IDEA-Research/GroundingDINO) via autodistill:
    https://github.com/autodistill/autodistill-grounding-dino
"""

from pathlib import Path
from typing import Literal, Sequence

import autodistill_grounding_dino
import numpy as np
import pandas as pd
import PIL
import supervision as sv
import tensorflow as tf
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from datasets import Dataset, load_metric
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import weighted_mode
from tqdm.notebook import tqdm
from transformers import (
    AutoImageProcessor,
    Dinov2ForImageClassification,
    Dinov2Model,
    Trainer,
    TrainingArguments,
)

from seegull.util.util import garbage_collect, split


class DINOv2Base:
    """Functions common to both the Huggingface and Tensorflow versions."""

    def create_embeddings_csv(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        additional_cols: Sequence[str] = (
            "image_key",
            "image_source",
            "brand",
            "object",
            "material",
        ),
    ) -> pd.DataFrame:
        """Create a standard format CSV given a df of images.

        Args:
            df: A DataFrame with the following columns:
                image_id,
                <columns required for loading an image: path or image_source>,
                x1, y1, x2, y2,
                <additional_cols>
            output_path: Where to save the CSV
            additional_cols: Additional columns from df to include in the CSV.
                If columns in `additional_cols` are not present in df, they
                will be silently ignored.
        """

        # TODO: Find a place to put the brand-specific logic
        # df = df.dropna(subset="brand")
        # df["brand"] = df["brand"].str.lower()
        # df = df[df["brand"] != "unknown"].reset_index(drop=True)
        df = df.copy()

        # Load the images
        df["image"] = self.load_images(df, return_type="PIL")

        # Remove missing images
        df = df.dropna(subset="image").reset_index(drop=True)

        # Get the embeddings for images
        df["embeddings"] = self.embeddings(images=df["image"].tolist())

        # Only keep the cols that exist in the dataframe
        additional_cols = [col for col in additional_cols if col in df.columns]

        # Make a copy with only the desired columns
        df = df[
            [
                "image_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "embeddings",
            ]
            + additional_cols
        ]

        # Save to CSV and return the DataFrame
        df.to_csv(output_path, index=False)
        return df

    def load_images(self, *args, **kwargs) -> list[PIL.Image.Image]:
        """Wrapper for `seegull.data.image.load_images configured for DINO."""
        from seegull.data.image import load_images

        return load_images(
            *args,
            crop=True,
            resize=(224, 224),
            # resize=1.0,
            # max_width=256,
            # max_height=256,
            **kwargs,
        ).tolist()


class DINOv2(DINOv2Base):
    """Wrapper around Huggingface Transformer's DINOv2 implementation"""

    def __init__(
        self,
        model_path: str | None = None,
        size: Literal["small", "base", "large", "giant"] = "small",
    ):
        """
        Args:
            model_path: If None, use the standard model in the given size.
                Otherwise, a path to a model on disk.
            size: One of the standard DINOv2 models from here:
                https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3
        """
        if model_path is None:
            model_path = f"facebook/dinov2-{size}"

        self.model_path = model_path
        self._processor = None
        self._model = None

    @property
    def processor(self) -> AutoImageProcessor:
        """Lazy-load the processor."""
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_path
            )

        return self._processor

    @property
    def model(self) -> Dinov2Model:
        """Lazy-load the model."""
        if self._model is None:
            self._model = Dinov2Model.from_pretrained(self.model_path)
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._model.to(device)

        return self._model

    @garbage_collect
    def embeddings(
        self,
        images: Sequence[PIL.Image.Image] | None = None,
        chunksize: int = 500,
        df: pd.DataFrame | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for the given images or dataframe.

        Args:
            images: A sequence of images in PIL.Image format.
            chunksize: How many images to process at a time
            df: Instead of passing images, a DataFrame in the standard seegull
                image DataFrame format can be passed and the images will be
                loaded from that.

        Returns:
            A list of embeddings, each of which is a list of floats.
        """
        if images is None:
            images = self.load_images(df, return_type="PIL")

        embeddings = []
        for chunk in tqdm(split(images, chunksize)):
            pixel_values = self.processor(
                images=chunk, return_tensors="pt"
            ).pixel_values.to(self.model.device)

            with torch.no_grad():
                outputs = self.model(pixel_values)

            emb = outputs.pooler_output
            embeddings = embeddings + emb.tolist()

        return embeddings

    def knn(self, df: pd.DataFrame, **kwargs) -> "KNNSearch":
        """Return a KNNSearch object for this embedding model.

        Args:
            df: The DataFrame to use as the embedding database.
        """
        return KNNSearch(self, df, **kwargs)

    @garbage_collect
    def finetune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target: str,
        output_dir: str,
    ):
        """Finetune the DINOv2 model through a classification task.

        Takes the DINOv2 model that this object was initialized with,
        adds a classification head and then trains the whole network to
        optimize for classification.

        Args:
            train_df: A DataFrame with the training split. Must have columns
                `pil_image` and `<target>`.
            train_df: A DataFrame with the validation split. Must have columns
                `pil_image` and `<target>`.
            target: The column of the DataFrames to train the model to predict.
            output_dir: The directory to write the new model to.
        """
        labels = sorted(train_df[target].unique())
        label2id = {c: i for i, c in enumerate(labels)}

        def gen_examples(split_df):
            for i, row in split_df.iterrows():
                yield {"pil_image": row.pil_image, "target": row[target]}

        train_ds = Dataset.from_generator(
            gen_examples, gen_kwargs={"split_df": train_df}
        )
        val_ds = Dataset.from_generator(
            gen_examples, gen_kwargs={"split_df": val_df}
        )

        model = Dinov2ForImageClassification.from_pretrained(
            self.model_path,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
        )

        def collate_fn(batch):
            pixel_values = self.processor(
                images=[x["pil_image"] for x in batch], return_tensors="pt"
            ).pixel_values

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor([label2id[x["target"]] for x in batch]),
            }

        metric = load_metric("accuracy")

        def compute_metrics(p):
            return metric.compute(
                predictions=np.argmax(p.predictions, axis=1),
                references=p.label_ids,
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            num_train_epochs=10,
            fp16=True,
            learning_rate=5e-6,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.processor,
        )

        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        return DINOv2(output_dir)


class KNNSearch:
    """Ties together an embedding model, a DataFrame of embeddings and search.

    KNNSearch takes a DataFrame as a "database" of embeddings and attributes
    and uses K-Nearest-Neighbors (KNN) search to find the closest rows to given
    inputs, or predict attributes by aggregating the closest rows.

    Also contains functions for evaluating the accuracy of an embedding model
    for similarity search.
    """

    def __init__(
        self,
        dino_model: DINOv2,
        df: pd.DataFrame,
        embeddings_col: str | None = None,
        image_col: str | None = None,
        n_neighbors: int = 10,
        max_distance: float | None = None,
    ):
        """
        Args:
            dino_model: The model to use for generating embeddings
            df: The DataFrame to use as the data backing the search.
                Requires either embeddings or images (to generate embeddings).
            embeddings_col: The column that contains the embeddings.
            image_col: The column that contains the images.
            n_neighbors: How many neighbors to find when searching.
            max_distance: If set, neighbors won't be returned if they're
                further than max_distance.
        """
        assert (
            embeddings_col or image_col
        ), "Either embeddings or images must be provided."

        self.dino_model = dino_model
        self.df = df.reset_index(drop=True)

        if embeddings_col is None:
            embeddings_col = "embeddings"
            self.df[embeddings_col] = self.dino_model.embeddings(
                self.df[image_col].tolist()
            )

        self.embeddings_col = embeddings_col

        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(np.vstack(self.df[embeddings_col]))
        self.max_distance = max_distance

    def search(
        self, embeddings: Sequence[np.array]
    ) -> list[list[tuple[float, pd.Series | None]]]:
        """Return the nearest neighbors for each embedding given.

        The return is a list of neighbors for each embedding. Each neighbor
        is a tuple of (distance, row) where row is the full row from self.df.

        Always returns n_neighbors results for each embedding.
        If self.max_distance is set, then any neighbor futher than max_distance
        will be replaced with (-1, None). This is done to keep the results
        a consistent shape for batch aggregation. If iterating through the
        results directly, simply ignore the results with distance == -1.

        Returns:
            A 2D list of tuples of (distance, row). The list has shape
                (len(embeddings), n_neighbors).
        """
        # Lookup the k nearest neighbors of each embedding
        # Returns a 2D array of distances and indicies
        # The shape of each is (len(embeddings), n_neighbors)
        dists, idxs = self.knn.kneighbors(np.vstack(embeddings))

        # Iterate through the results to create the desired return format
        # described above.
        return [
            [
                # Return a tuple of (distance, row)
                # If max_distance is configure or distance is greater than
                # max_distance then return (-1, None)
                (d, r)
                if (not self.max_distance) or (d <= self.max_distance)
                else (-1, None)
                # Iterate through pairs of distance and row
                # The rows come by doing a lookup on self.df based
                # on the array of indicies returned above.
                for d, r in zip(
                    emb_d,
                    self.df.iloc[emb_i]
                    .drop(self.embeddings_col, axis=1)
                    .itertuples(index=True),
                )
            ]
            # Iterate through each set of distances and indicies
            # for each of the given embeddings. `emb_d` and `emb_i` both
            # have length `n_neighbors`.
            for emb_d, emb_i in zip(dists, idxs)
        ]

    def predict(
        self,
        df: pd.DataFrame,
        embeddings_col: str,
        target_cols: str | Sequence[str],
        index_col: str | None = None,
        weighted: bool = True,
    ) -> pd.DataFrame:
        """Predict each of the target cols by aggegating the nearest neighbors.

        Given a DataFrame with an embeddings column, find the nearest neighbors
        for each row. Then use those rows to vote on each of the target_cols.
        If weighted=True, the votes are weighted by the inverse of the distance,
        otherwise all votes are given equal weight.

        Return the results as a DataFrame with two columns per target column:
        the prediction for that column, and a score, which is the (weighted)
        proportion of votes for the winning prediction.

        Args:
            df: A DataFrame with embeddings_col and optionally an
                index_col (described below).
            embeddings_col: The column containing embeddings
            target_cols: The column(s) from self.df to predict
            index_col: A column representing a unique ID, such as `image_id`.
                If this is set, any rows in df with the same unique ID as a
                matched row in self.df won't be included in the prediction.
                This makes it possible to predict for an image included in
                self.df without including itself in the prediction.
            weighted: Whether to weight the votes by distance

        Returns:
            A DataFrame with the predictions and score for each of target_cols.
        """
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        # Get the raw search results
        results = self.search(df[embeddings_col])

        # Filter the results to remove neighbors with matching `index_col`.
        # And convert each result to the format
        # [target_col_0, ..., target_col_n, 1/dist]
        results_filtered = []
        for i, nearest in enumerate(results):
            filtered = []
            for dist, row in nearest:
                if (row is not None) and (
                    not index_col
                    or (
                        getattr(df.iloc[i], index_col)
                        != getattr(row, index_col)
                    )
                ):
                    filtered.append(
                        [getattr(row, t) if row else "" for t in target_cols]
                        # Use the inverse of the distance of weighted
                        + [1 / dist if weighted else 1]
                        # + [-1 if dist == -1 else 1 / dist if weighted else 1]
                    )
                else:
                    # If using index_col and the index_cols match or if the row
                    # is None (because it was too far away) include an "unknown"
                    # result. This is given 0 weight but is needed to keep the
                    # shape of each result the same.
                    filtered.append(["unknown" for t in target_cols] + [0])
            results_filtered.append(filtered)

        matches = np.array(results_filtered)

        output_cols = {}
        for i, target_col in enumerate(target_cols):
            # Aggregate each of the targets using weighted_mode.
            # This returns both the predictions and the sum of the
            # weights for the prediction.
            # This operation reduces the shape from
            # (len(df), n_neighbors, 1) -> (len(df),)
            # for both pred and weights.
            pred, weights = weighted_mode(
                matches[:, :, i], matches[:, :, -1], axis=1
            )

            # Normalize the weights by the sum of all weights
            conf = weights.flatten() / np.clip(
                matches[:, :, -1].astype(float), 0, None
            ).sum(axis=1)

            output_cols[target_col] = pred.flatten()
            output_cols[f"{target_col}_conf"] = conf

        output_df = pd.DataFrame(output_cols)
        output_df.index = df.index
        return output_df

    def evaluate_predict(
        self,
        df: pd.DataFrame,
        embeddings_col: str,
        target_cols: str | Sequence[str],
        prefix: str = "",
        **kwargs,
    ) -> pd.DataFrame:
        """Write the results of KNNSearch.predict to df in-place.

        Write three columns to df in-place for each target:
        _ {prefix}_{target_col}_pred: The predicted value of the target
        - {prefix}_{target_col}_correct: Whether the targets match
        - {prefix}_{target_col}_conf: The confidence of the prediction

        Args:
            prefix: A string to prefix each new column with
            (See KNNSearch.predict for description of other args)
        """
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        if prefix and not prefix.endswith("_"):
            prefix = prefix + "_"

        predictions = self.predict(df, embeddings_col, target_cols, **kwargs)

        for col in target_cols:
            df[f"{prefix}{col}"] = predictions[col]
            df[f"{prefix}{col}_conf"] = predictions[f"{col}_conf"]
            df[f"{prefix}{col}_correct"] = df[col] == predictions[col]

        return df


# TODO: Move this elsewhere since it's not specific to DINO
# def metrics(
#     df: pd.DataFrame,
#     groupby: str,
#     count_col: str,
#     mean_cols: Sequence[str],
#     verbose=True,
# ) -> pd.DataFrame:
#     agg_df = (
#         df.groupby(groupby)
#         .agg({col: "mean" for col in mean_cols} | {count_col: "count"})
#         .sort_values(count_col, ascending=False)
#     )

#     if verbose:
#         for col in mean_cols:
#             print(
#                 col,
#                 (agg_df[col] * agg_df[count_col]).sum()
#                 / agg_df[count_col].sum(),
#             )

#     return agg_df.rename(columns={count_col: "count"})

# Standard normalization factors used by pytorch/huggingface
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class DINOv2TF(DINOv2Base):
    """A Wrapper for a Tensorflow version of DINOv2.

    The Tensorflow version comes from converting the onnx models shared here:
    https://github.com/facebookresearch/dinov2/issues/19#issuecomment-1880139201

    This wrapper is primarily used for adding a new signature on top of the
    basic Tensorflow version that includes the same preprocessing steps that
    Huggingface's DINOv2 has. The signature is called "with_preprocessing".

    The wrapper also exposes a function to generate embeddings using the TF
    model. If the TF model includes the "with_preprocessing" siganture, it will
    use that for generating embeddings, otherwise it will use the
    "serving_default" signature.
    """

    def __init__(self, path: str | Path):
        """Loads the model from a Tensorflow saved_model format."""
        self.path = Path(path)
        self.model = tf.saved_model.load(self.path)
        self.has_preprocessing = "with_preprocessing" in self.model.signatures
        self.signature = (
            self.model.signatures["with_preprocessing"]
            if self.has_preprocessing
            else self.model.signatures["serving_default"]
        )

    def add_preprocessing_signature(self, resize=256, crop_size=224):
        """Add a new signature to the model that includes preprocessing steps.

        Args:
            resize: The size to change the input image to. Defaults to256x256.
            crop_size: The size of the center crop to take after the inital
                resize. Defaults to 224x224.
        """
        if self.has_preprocessing:
            print("The model already has a preprocessing siganture!")
            return

        @tf.function
        def with_preprocessing(image):
            # Resize the image to the shortest edge as 256px
            shape = tf.cast(tf.shape(image), tf.float32)
            min_side = tf.minimum(shape[0], shape[1])
            if min_side < 1:
                return {
                    "embedding": tf.zeros(
                        self.signature.output_shapes["output"][-1]
                    )
                }
            ratio = min_side / resize
            resize_height, resize_width = (
                tf.clip_by_value(
                    tf.cast(shape[0] / ratio, tf.int32), 1, tf.int32.max
                ),
                tf.clip_by_value(
                    tf.cast(shape[1] / ratio, tf.int32), 1, tf.int32.max
                ),
            )
            i = tf.image.resize(
                image,
                [resize_height, resize_width],
                method="bicubic",
                antialias=True,
            )

            # Round to nearest int
            i = tf.round(i)

            # Crop to 224x224 in the center of the image
            y_margin = (resize_height - crop_size) // 2
            x_margin = (resize_width - crop_size) // 2
            i = i[
                y_margin : y_margin + crop_size, x_margin : x_margin + crop_size
            ]

            # Rescale
            i = i / 255.0

            # Normalize
            i = tf.math.divide(
                tf.math.subtract(i, IMAGENET_DEFAULT_MEAN), IMAGENET_DEFAULT_STD
            )

            # Calculate embeddings using dino
            i = tf.expand_dims(i, 0)
            i = self.signature(input=i)["output"]

            # Flatten and return
            return {"embedding": tf.reshape(i, [-1])}

        with_preprocessing_signature = with_preprocessing.get_concrete_function(
            image=tf.TensorSpec(
                tf.TensorShape([None, None, 3]), dtype=tf.uint8, name="image"
            )
        )

        output_path = self.path / "with_preprocessing"

        print("Writing new model to", output_path)

        tf.saved_model.save(
            self.model,
            output_path,
            signatures={
                "serving_default": self.signature,
                "with_preprocessing": with_preprocessing_signature,
            },
        )

        return DINOv2TF(output_path)

    def embeddings(self, df=None):
        embeddings = []

        images = self.load_images(df)

        for image in tqdm(images):
            try:
                embeddings.append(
                    self.signature(image=tf.reverse(image.image, axis=[2]))[
                        "embedding"
                    ]
                    .numpy()
                    .tolist()
                )
            except:
                embeddings.append(None)

        return embeddings


class AutodistillModelWrapper:
    """A wrapper for autodistill models to have a common predict interface"""

    import seegull

    def set_model(
        self, model: DetectionBaseModel, ontology: dict[str, str], **kwargs
    ):
        self.model = model(
            ontology=CaptionOntology(ontology),
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        # Save the default thresholds to restore after prediction
        self.default_box_threshold = self.model.box_threshold
        self.default_text_threshold = self.model.text_threshold

    @property
    def ontology(self):
        return self.model.ontology

    @ontology.setter
    def ontology(self, ontology):
        self.model.ontology = CaptionOntology(ontology)

    @property
    def classes(self):
        return self.model.ontology.classes()

    def predict(
        self, image: "seegull.data.image.Image", conf: float | None = None
    ) -> sv.Detections:
        """Detect objects for the given image.

        Args:
            image: The image to detect objects for
            conf: The confidence to use for both box and text threshold.
                If None, use the model's defaults.
        """
        # Autodistill models take a confidence threshold at model creation time
        # but instead we pass it at prediction time. This sets the thresholds.
        if conf is not None:
            self.model.box_threshold = conf
            self.model.text_threshold = conf

        result = self.model.predict(image.to_pil())

        # Restore the original thresholds. This isn't really necessary the way
        # we're using it.
        self.model.box_threshold = self.default_box_threshold
        self.model.text_threshold = self.default_text_threshold

        return result


class GroundingDINO(AutodistillModelWrapper):
    @garbage_collect
    def __init__(self, ontology: dict[str, str], **kwargs):
        self.set_model(
            autodistill_grounding_dino.GroundingDINO, ontology, **kwargs
        )
