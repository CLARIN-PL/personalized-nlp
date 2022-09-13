from typing import List
import os
from pathlib import Path

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class ToxicityCartographyRegressorDataModule(BaseDataModule):
    
    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "wiki_data"
    
    @property
    def worker_demographics_file(self) -> str:
        return None
    
    @property 
    def annotations_file(self) -> str:
        return f'{self.model_name}_regressor.csv'
    
    @property 
    def data_file(self) -> str:
        return f'toxicity_annotated_comments_processed.csv'

    @property
    def class_dims(self) -> List[int]:
        return [1]

    @property
    def annotation_columns(self) -> List[str]:
        return [self.value]

    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir / f"embeddings/rev_id_to_emb_{self.embeddings_type}_toxic.p"
        )

    def __init__(
        self, model: str, value: str, **kwargs,
    ):
        self.model_name = model
        self.value = value
        super().__init__(**kwargs)
        # os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def _remap_column_names(self, df):
        mapping = {"rev_id": "text_id", "worker_id": "annotator_id", "comment": "text"}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / self.data_file
        )
        self.data = self._remap_column_names(self.data)
        self.data["text"] = self.data["text"].str.replace("NEWLINE_TOKEN", "  ")

        # self.annotators = pd.read_csv(
        #     self.data_dir / self.worker_demographics_file, sep='\t'
        # )
        # self.annotators = self._remap_column_names(self.annotators)

        self.annotations = pd.read_csv(
            self.data_dir / self.annotations_file
        )
        self.annotations = self._remap_column_names(self.annotations)
