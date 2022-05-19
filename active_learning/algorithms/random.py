import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase


class RandomSelector(TextSelectorBase):
    def select_annotations(
        self,
        texts: pd.DataFrame,
        amount: int,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        return not_annotated.sample(n=amount)
