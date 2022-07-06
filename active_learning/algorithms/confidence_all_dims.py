import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase
from scipy.spatial.distance import cosine


class ConfidenceAllDimsSelector(TextSelectorBase):
    def select_annotations(
        self,
        texts: pd.DataFrame,
        amount: int,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        if confidences is not None:
            non_confident_vector = 0.5 * np.ones_like(confidences[0])

            cosine_distances = [cosine(c, non_confident_vector) for c in confidences]
            sorted_index = np.argsort(cosine_distances)

            return not_annotated.iloc[sorted_index[:amount]]

        amount = min(amount, len(not_annotated.index))
        return not_annotated.sample(n=amount)
