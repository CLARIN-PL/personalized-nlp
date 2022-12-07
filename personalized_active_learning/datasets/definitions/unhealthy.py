import os
from typing import List, Tuple

import pandas as pd

from personalized_active_learning.datasets import BaseDataModule
from personalized_active_learning.datasets.base import SplitMode
from settings import DATA_DIR


class UnhealthyDataModule(BaseDataModule):
    @property
    def annotations_file_relative_path(self) -> str:
        # TODO: Leftover of previous code
        if self.split_mode == SplitMode.TEXTS:
            stratification_type = "texts"
        elif self.split_mode == SplitMode.USERS:
            stratification_type = "users"
        else:
            raise Exception(f"Unsupported split mode {self.split_mode.value}")
        return f"uc_annotations_{stratification_type}_folds.csv"

    @property
    def annotation_columns(self) -> List[str]:
        return [
            "antagonize",
            "condescending",
            "dismissive",
            "generalisation",
            "generalisation_unfair",
            "healthy",
            "hostile",
            "sarcastic",
        ]

    @property
    def classes_dimensions(self) -> List[int]:
        return [2] * 8

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def load_data_and_annotations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.data_dir = DATA_DIR / "unhealthy_conversations"
        columns_map = {"comment": "text"}
        data = pd.read_csv(self.data_dir / "uc_texts_processed.csv").dropna()
        data = data.rename(columns=columns_map)

        annotations = pd.read_csv(self.data_dir / self.annotations_file_relative_path)

        annotations = annotations.drop_duplicates(subset=["text_id", "annotator_id"])
        return data, annotations
