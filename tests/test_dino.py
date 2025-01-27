import numpy as np

from seegull import DINOv2


def test_dino_runs(annotation_df):
    # Generate embeddings
    dinov2 = DINOv2()

    # Create a unique ID for each row/annotation
    annotation_df["id"] = np.arange(len(annotation_df))
    annotation_df["embeddings"] = dinov2.embeddings(df=annotation_df)

    # Initialize a KNNSearch object with this model and data
    knn = dinov2.knn(annotation_df, n_neighbors=5)

    knn.evaluate_predict(annotation_df, "object", index_col="id")[
        ["object", "object_pred", "object_conf", "object_correct"]
    ]
