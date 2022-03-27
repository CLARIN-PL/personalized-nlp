from typing import List

import pandas as pd
import os

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class HumorDataModule(BaseDataModule):
    @property
    def embeddings_path(self) -> Path:
        return STORAGE_DIR / f"humor/embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def annotation_columns(self) -> List[str]:
        return ["is_funny"]

    @property
    def class_dims(self):
        return [2]

    def __init__(
        self,
        min_annotations_per_text=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / "humor/"

        self.min_annotations_per_text = min_annotations_per_text

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / "texts" / "data.csv")

        self.annotations = pd.read_csv(
            self.data_dir / "texts" / "annotations.csv"
        ).dropna()

        if self.min_annotations_per_text is not None:
            text_id_value_counts = self.annotations.text_id.value_counts()
            text_id_value_counts = text_id_value_counts[
                text_id_value_counts >= self.min_annotations_per_text
            ]
            self.annotations = self.annotations.loc[
                self.annotations.text_id.isin(text_id_value_counts.index.tolist())
            ]
