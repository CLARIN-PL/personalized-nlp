import os
from pathlib import Path
from typing import List

import pandas as pd

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class CockamamieGobbledegookDataModule(BaseDataModule):
    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "cockamamie_gobbledegook"

    @property
    def annotation_columns(self) -> List[str]:
        return ["is_funny"]

    @property
    def class_dims(self):
        return [2]

    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / "texts" / "cockamamie_gobbledegook_texts_e.csv"
        )
        self.data["text"] = self.data["text_english"]

        self.annotations = pd.read_csv(
            self.data_dir
            / "texts"
            / "cockamamie_annotations_only_controversial_a_non_empty.csv"
        ).dropna()
