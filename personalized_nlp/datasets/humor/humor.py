from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class HumorDataModule(BaseDataModule):
    
    @property 
    def annotations_file(self) -> str:
        return f'annotations_{self.stratify_folds_by}_folds.csv'
    
    @property 
    def data_file(self) -> str:
        return f'data_processed.csv'
    
    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "humor"

    @property
    def annotation_columns(self) -> List[str]:
        return ["is_funny"]

    @property
    def class_dims(self):
        return [6]

    def __init__(
        self, min_annotations_per_text=None, **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_annotations_per_text = min_annotations_per_text

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / "texts" / self.data_file)

        self.annotations = pd.read_csv(
            self.data_dir / "texts" / self.annotations_file
        ).dropna()

        if self.min_annotations_per_text is not None:
            text_id_value_counts = self.annotations.text_id.value_counts()
            text_id_value_counts = text_id_value_counts[
                text_id_value_counts >= self.min_annotations_per_text
            ]
            self.annotations = self.annotations.loc[
                self.annotations.text_id.isin(text_id_value_counts.index.tolist())
            ]
