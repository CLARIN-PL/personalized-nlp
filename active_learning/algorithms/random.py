from typing import List
import pandas as pd
import numpy as np


def random_selector(
    texts: pd.DataFrame,
    amount: int,
    annotated: pd.DataFrame,
    not_annotated: pd.DataFrame,
    confidences: np.ndarray,
):
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

    return not_annotated.sample(n=amount)
