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
    ) -> None:
        super().__init__(class_dims, annotation_columns, amount_per_user)
        self.regressor = None

    def get_metrics(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ):
        text_counts_df = annotated.text_id.value_counts().reset_index()
        text_counts_df.columns = ["text_id", "text_count"]

        sampled_annotations = not_annotated.merge(text_counts_df, how="left").fillna(
            100
        )

        return sampled_annotations["text_count"].values[-1, None]

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
