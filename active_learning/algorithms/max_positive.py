import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase
from scipy.spatial.distance import cosine


class MaxPositiveClassSelector(TextSelectorBase):
    def select_annotations(
        self,
        texts: pd.DataFrame,
        amount: int,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        if confidences is not None:
            dims = confidences.shape[1]
            confidences = confidences.reshape(-1, int(dims / 2), 2)[:, :, 1]
            confidences = confidences.mean(axis=1)

            sorted_index = np.argsort(confidences)[::-1]

            return not_annotated.iloc[sorted_index[:amount]]

        amount = min(amount, len(not_annotated.index))
        return not_annotated.sample(n=amount)
