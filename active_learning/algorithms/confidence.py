from typing import Optional
import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase
from scipy.spatial.distance import cosine


class ConfidenceSelector(TextSelectorBase):
    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if confidences is not None:
            confidences = confidences.max(axis=1)
            not_annotated["confidence"] = np.argsort(confidences)
            not_annotated = not_annotated.sort_values(by="confidence")
        else:
            not_annotated = not_annotated.sample(frac=1.0)

        return not_annotated


class Confidencev2Selector(TextSelectorBase):
    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        not_annotated_original = not_annotated.copy()
        if confidences is not None:
            confidences = confidences.max(axis=1)
            not_annotated["confidence"] = confidences.tolist()

            not_annotated = not_annotated.sort_values(by="confidence")
            not_annotated = not_annotated.groupby("text_id").apply(
                lambda rows: rows[:1]
            )
            not_annotated = not_annotated.sort_values(by="confidence")
            not_annotated = not_annotated.reset_index(drop=True)
            not_annotated = not_annotated.loc[:, ["text_id", "annotator_id"]]

            result_df = not_annotated_original.merge(
                not_annotated, on=["text_id", "annotator_id"]
            )
            result_df = result_df.set_index("original_index")

            return result_df

        return not_annotated.sample(frac=1.0)


class ConfidenceAllDimsSelector(TextSelectorBase):
    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
        amount: Optional[int] = None,
        amount_per_user: Optional[int] = None,
    ) -> pd.DataFrame:
        if confidences is not None:
            non_confident_vector = 0.5 * np.ones_like(confidences[0])

            cosine_distances = [cosine(c, non_confident_vector) for c in confidences]
            sorted_index = np.argsort(cosine_distances)

            return not_annotated.iloc[sorted_index]

        return not_annotated.sample(frac=1.0)
