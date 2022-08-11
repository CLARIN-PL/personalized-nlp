from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd

from active_learning.algorithms.utils import stratify_by_users


class TextSelectorBase(ABC):
    def __init__(
        self,
        class_dims: Optional[List[int]] = None,
        annotation_columns: Optional[List[str]] = None,
        amount_per_user: Optional[int] = None,
    ) -> None:
        self.class_dims = class_dims
        self.annotation_columns = annotation_columns
        self.amount_per_user = amount_per_user

    def select_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
        amount: Optional[int] = None,
    ) -> pd.DataFrame:
        """Selects `amount` annotations (or `amount_per_users` per each user) annotations from `not_annotated` dataframe.
        You can't change the index of `not_annotated` subrows as they are used as identificator of annotation.

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
        amount_per_user = self.amount_per_user
        if amount is None and amount_per_user is None:
            raise ValueError("Either amount or amount_per_user must be not None")

        sorted_annotations = self.sort_annotations(
            texts, annotated, not_annotated, confidences
        )

        if amount_per_user is not None:
            sorted_annotations = stratify_by_users(sorted_annotations, amount_per_user)
        else:
            sorted_annotations = sorted_annotations[:amount]

        return sorted_annotations

    @abstractmethod
    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        pass
