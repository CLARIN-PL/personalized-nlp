import os

import pandas as pd

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule
from typing import List
from pathlib import Path


class MeasuringHateSpeechDataModule(BaseDataModule):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / "measuring_hate_speech"

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def _remap_column_names(self, df):
        mapping = {
            "comment_id": "text_id",
            "worker_id": "annotator_id",
            "comment": "text"
        }
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / ("data.tsv"),
            sep="\t",
        )
        self.data = self._remap_column_names(self.data)
        # self.data["text"] = self.data["text"].str.strip('"')

        self.annotators = pd.read_csv(
            self.data_dir / ("worker_demographics.tsv"),
            sep="\t",
        )
        self.annotators = self._remap_column_names(self.annotators)

        self.annotations = pd.read_csv(self.data_dir / ("annotations.tsv"),
                                       sep="\t")
        self.annotations = self._remap_column_names(self.annotations)

    @property
    def class_dims(self) -> List[int]:
        return [5, 5, 5, 5, 5, 5, 5, 5, 5, 3]

    @property
    def annotation_columns(self) -> List[str]:
        return [
            "sentiment", "respect", "insult", "humiliate", "status",
            "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"
        ]

    @property
    def embeddings_path(self) -> Path:
        return (
            STORAGE_DIR /
            f"measuring_hate_speech/embeddings/rev_id_to_emb_{self.embeddings_type}.p"
        )