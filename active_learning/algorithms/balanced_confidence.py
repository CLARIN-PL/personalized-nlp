import warnings

import pandas as pd
import numpy as np

from active_learning.algorithms.base import TextSelectorBase


class BalancedConfidenceSelector(TextSelectorBase):
    def __init__(self, select_minimal_texts: bool = True, *args, **kwargs) -> None:
        """Selector based on model confidence balance between classes.
        If there is a conflict, it is resolved based on absolute difference between model confidences between classes for a specific task.
        If there are more than one task then the average is calculated across all tasks

        """
        super(BalancedConfidenceSelector, self).__init__()
        self.select_minimal_texts: bool = select_minimal_texts

    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        """Select annotations using rule, described in __init__()

        Args:
            texts (pd.DataFrame): Dataframe with texts data. It contains (at least)
            two columns: `text_id` (int) and `text` (str)
            amount (int): Number of annotations to select from `not_annotated` dataframe.
            annotated (pd.DataFrame): Already selected annotations. It is a Dataframe with
            (at least) two columns: `text_id` and `annotator_id`.
            not_annotated (pd.DataFrame): Annotations to select. It is a Dataframe with
            (at least) two columns: `text_id` and `annotator_id`. The algorithm should return
            `amount` of rows from this dataframe.
            confidences (np.ndarray): Numpy array of prediction confidences for all annotations
            in `not_annotated` dataframe. It has length equal to length of `not_annotated` dataframe.

        Returns:
            pd.DataFrame: Dataframe with subset of rows of `not_annotated` dataframe with length equal to
            `amount`.
        """
        if confidences is not None:
            dims = confidences.shape[1]
            print(f"INFO: confidences shape: {confidences.shape}")
            confidences = confidences.reshape(-1, int(dims / 2), 2)
            confidences = np.absolute(np.diff(confidences, axis=2))
            confidences = confidences.mean(axis=1)
            confidences = confidences.flatten()

            sorted_index = np.argsort(confidences)

            return not_annotated.iloc[sorted_index]

        warnings.warn(
            f"There is no confidences, sampled of samples from not annotated data."
        )

        return not_annotated.sample(frac=1.0)
