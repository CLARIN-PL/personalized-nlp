from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class UnhealthyDataModule(BaseDataModule):
    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "unhealthy_conversations"

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
        columns_map = {'comment': 'text'}
        self.data = pd.read_csv(self.data_dir / "uc_texts_processed.csv").dropna() 
        self.data = self.data.rename(columns=columns_map)

        self.annotations = pd.read_csv(self.data_dir / f"uc_annotations_{self.stratify_folds_by}_folds.csv")
        
