import os

from pathlib import Path
from typing import List

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class ClarinEmoTextDataModule(BaseDataModule):

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "clarin_emo_text"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def _remap_column_names(self, df):
        mapping = {"user_id": "annotator_id"}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / "data.tsv",
            sep="\t",
        )
        self.data = self._remap_column_names(self.data)

        self.annotations = pd.read_csv(self.data_dir / "annotations.tsv",
                                       sep="\t")
        self.annotations = self._remap_column_names(self.annotations)

    @property
    def class_dims(self) -> List[int]:
        return [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    @property
    def annotation_columns(self) -> List[str]:
        return [
            'radość', 'zaufanie', 'przeczuwanie', 'zdziwienie', 'strach',
            'smutek', 'wstręt', 'gniew', 'pozytywny', 'negatywny', 'neutralny'
        ]

    @property
    def embeddings_path(self) -> Path:
        return (self.data_dir /
                f"embeddings/rev_id_to_emb_{self.embeddings_type}.p")
