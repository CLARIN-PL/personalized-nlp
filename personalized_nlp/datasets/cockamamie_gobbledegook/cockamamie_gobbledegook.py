import os
from pathlib import Path
from typing import List

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class CockamamieGobbledegookDataModule(BaseDataModule):
    @property
    def annotations_file(self) -> str:
        return f"cockamamie_annotations_only_controversial_a_non_empty_{self.stratify_folds_by}_folds.csv"

    @property
    def data_file(self) -> str:
        return "cockamamie_texts_only_controversial_a_non_empty_processed.csv"


    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "cockamamie_gobbledegook"

    @property
    def annotation_columns(self) -> List[str]:
        return ["is_funny"]

    @property
    def class_dims(self):
        return [2]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir /  self.data_file)
        self.data["text"] = self.data["text_english"]

        self.annotations = pd.read_csv(
            self.data_dir /  self.annotations_file
        ).dropna()
