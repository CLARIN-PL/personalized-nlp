from typing import Optional
import pandas as pd
import numpy as np
from active_learning.algorithms.base import TextSelectorBase

from itertools import cycle


class TextAnnotationDiversitySelector(TextSelectorBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.text_ids_cycle = None

    def sort_annotations(
        self,
        texts: pd.DataFrame,
        annotated: pd.DataFrame,
        not_annotated: pd.DataFrame,
        confidences: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        text_ids_unique = texts.sample(frac=1.0).text_id.drop_duplicates().to_frame()

        if self.text_ids_cycle is None:
            self.text_ids_cycle = cycle(text_ids_unique["text_id"].tolist())

        text_id_indexes = not_annotated.groupby("text_id").groups

        added_indexes = []
        added_number = 0
        for text_id in self.text_ids_cycle:
            if text_id not in text_id_indexes or len(text_id_indexes[text_id]) == 0:
                continue

            annotation_index = text_id_indexes[text_id][0]
            text_id_indexes[text_id] = text_id_indexes[text_id][1:]

            added_indexes.append(annotation_index)

            added_number = len(added_indexes)
            if added_number == len(not_annotated.index):
                break

        return not_annotated.loc[added_indexes].sample(frac=1.0)
