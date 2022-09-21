from ast import Not
from typing import List, Optional
import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase


class ReinforceSelector(TextSelectorBase):
    def __init__(
        self,
        class_dims: Optional[List[int]] = None,
        annotation_columns: Optional[List[str]] = None,
        amount_per_user: Optional[int] = None,
        text_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(class_dims, annotation_columns, amount_per_user)
        self.regressor = None
        self.text_embeddings = text_embeddings

    def get_metrics(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ):
        cols = ["text_count", "annotator_count", "max_confidence"]
        if confidences is not None:
            cols += [f"conf_{i}" for i in range(confidences.shape[1])]

        text_counts_df = annotated.text_id.value_counts().reset_index()
        text_counts_df.columns = ["text_id", "text_count"]

        annotator_counts_df = annotated.annotator_id.value_counts().reset_index()
        annotator_counts_df.columns = ["annotator_id", "annotator_count"]

        not_annotated = not_annotated.merge(text_counts_df, how="left").fillna(0)
        not_annotated = not_annotated.merge(annotator_counts_df, how="left").fillna(0)

        if confidences is not None:
            not_annotated["max_confidence"] = confidences.argmax(axis=1)
            for i in range(confidences.shape[1]):
                not_annotated[f"conf_{i}"] = confidences[:, i]
        else:
            not_annotated["max_confidence"] = 0

        metrics = not_annotated.loc[:, cols].values
        if self.text_embeddings is not None:
            text_id_to_idx = {id: idx for id, idx in zip(texts.index, texts.text_id)}
            embeddings = np.vstack(
                [self.text_embeddings[text_id_to_idx[i]] for i in not_annotated.text_id]
            )
            metrics = np.hstack([metrics, embeddings])

        num_done_annotations = len(annotated.index)
        num_done_annotations_arr = np.ones((metrics.shape[0], 1)) * num_done_annotations
        metrics = np.hstack([metrics, num_done_annotations_arr])

        return metrics

    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if self.regressor is None:
            return not_annotated.sample(frac=1.0)

        metrics = self.get_metrics(
            texts=texts,
            annotated=annotated,
            not_annotated=not_annotated,
            confidences=confidences,
        )

        scores = self.regressor.predict(metrics)

        return not_annotated.iloc[np.argsort(scores)[::-1]]
