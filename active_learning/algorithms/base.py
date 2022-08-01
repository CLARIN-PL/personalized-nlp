from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd


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

    @abstractmethod
    def select_annotations(
        self,
        texts: pd.DataFrame,
        amount: int,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ) -> pd.DataFrame:
        """Baseline active learning algorithm, which selects random annotations.

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
        pass
