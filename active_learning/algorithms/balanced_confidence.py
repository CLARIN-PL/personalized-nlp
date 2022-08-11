import warnings

import pandas as pd
import numpy as np

from typing import Optional

from active_learning.algorithms.base import TextSelectorBase


class BalancedConfidenceSelector(TextSelectorBase):

    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
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
            print(f'INFO: confidences shape: {confidences.shape}')
            confidences = confidences.reshape(-1, int(dims / 2), 2)
            confidences = np.absolute(np.diff(confidences, axis=2))
            confidences = confidences.mean(axis=1)
            confidences = confidences.flatten()

            sorted_index = np.argsort(confidences)

            return not_annotated.iloc[sorted_index]

        return not_annotated.sample(frac=1.0)
