import os
from pathlib import Path

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class WikiDataModule(BaseDataModule):
    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "wiki_data"

    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def _remap_column_names(self, df):
        mapping = {"rev_id": "text_id", "worker_id": "annotator_id", "comment": "text"}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / (self.annotation_columns[0] + "_annotated_comments.tsv"),
            sep="\t",
        )
        self.data = self._remap_column_names(self.data)
        self.data["text"] = self.data["text"].str.replace("NEWLINE_TOKEN", "  ")

        self.annotators = pd.read_csv(
            self.data_dir / (self.annotation_columns[0] + "_worker_demographics.tsv"),
            sep="\t",
        )
        self.annotators = self._remap_column_names(self.annotators)

        self.annotations = pd.read_csv(
            self.data_dir / (self.annotation_columns[0] + "_annotations.tsv"), sep="\t"
        )
        self.annotations = self._remap_column_names(self.annotations)
