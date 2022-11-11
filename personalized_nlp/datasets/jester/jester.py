from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class JesterDataModule(BaseDataModule):
    
    @property 
    def annotations_file(self) -> str:
        return f'jester_annotations_{self.stratify_folds_by}_folds.csv'
    
    @property 
    def data_file(self) -> str:
        return f'data_processed.csv'
    
    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "jester"

    @property
    def annotation_columns(self) -> List[str]:
        return ["humor"]

    @property
    def class_dims(self):
        return [2]

    def __init__(
        self, binarize=False, **kwargs,
    ):
        super().__init__(**kwargs)

        self.binarize = binarize

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file)

        self.annotations = pd.read_csv(
            self.data_dir / self.annotations_file
        ).dropna()
        if self.binarize:
            self.binarize_labels()

    def binarize_labels(self):
        annotation_column = self.annotation_columns[0]
        df = self.annotations

        df[annotation_column] = (
            df[annotation_column].apply(lambda x: 1 if x > 6 else 0).astype(int)
        )
        self.annotations = df
