from typing import List

import pandas as pd
import os
from pathlib import Path

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class UnhealthyDataModule(BaseDataModule):
    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return STORAGE_DIR / "unhealthy_conversations"

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
    def class_dims(self):
        return [2] * 8

    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        full_data = pd.read_csv(self.data_dir / "unhealthy_full.csv").dropna()
        self.data = (
            full_data.loc[:, ["_unit_id", "comment"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.data.columns = ["text_id", "text"]

        self.annotations = full_data.loc[
            :, ["_unit_id", "_worker_id"] + self.annotation_columns
        ]
        self.annotations.columns = ["text_id", "annotator_id"] + self.annotation_columns
