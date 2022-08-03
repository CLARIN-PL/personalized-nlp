from typing import Optional
import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase


class RandomSelector(TextSelectorBase):
    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        return not_annotated.sample(frac=1.0)


class RandomImprovedSelector(TextSelectorBase):
    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: np.ndarray,
    ):
        if len(annotated.index) > 15_000:
            selected_texts = annotated.text_id.unique()
            not_annotated = not_annotated.loc[
                not_annotated.text_id.isin(selected_texts)
            ]

            if len(self.annotation_columns) == 1:
                col = self.annotation_columns[0]

                counts = annotated.loc[:, ["text_id", col]].value_counts()

                entropies = counts.reset_index().groupby("text_id")[0].apply(entropy)
                texts_with_entropy = entropies[entropies > 0.05].index.tolist()

                not_annotated = not_annotated.loc[
                    not_annotated.text_id.isin(texts_with_entropy)
                ]

        return not_annotated.sample(frac=1.0)
