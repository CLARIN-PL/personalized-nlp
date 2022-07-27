from __future__ import annotations
from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class EmotionsSimpleDataModule(BaseDataModule):
    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir
            / f"embeddings/text_id_to_emb_{self.embeddings_type}_{self.language}.p"
        )

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "emotions_simple_data"


    @property 
    def annotations_file(self) -> str:
        return f'cawi2_annotations2_{self.stratify_folds_by}_folds.csv'
    
    @property 
    def data_file(self) -> str:
        return f'cawi2_texts_multilang_processed.csv'

    @property
    def annotation_columns(self) -> List[str]:
        return [
            "OCZEKIWANIE",
            "POBUDZENIE EMOCJONALNE",
            "RADOŚĆ",
            "SMUTEK",
            "STRACH",
            "WSTRĘT",
            "ZASKOCZENIE",
            "ZAUFANIE",
            "ZNAK EMOCJI",
            "ZŁOŚĆ",
        ]

    @property
    def class_dims(self):
        return [5] * 8 + [7, 5]

    def __init__(
        self, language: str = "english", **kwargs,
    ):
        self.language = language
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file)
        self.data.loc[:, "text"] = self.data.loc[:, "text_" + self.language]

        self.annotations = pd.read_csv(self.data_dir / self.annotations_file).dropna()
        self.annotators = pd.read_csv(self.data_dir / self.annotations_file)
