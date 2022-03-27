import os
from pathlib import Path

import pandas as pd

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class CockamamieGobbledegookDataModule(BaseDataModule):
    @property
    def embeddings_path(self) -> Path:
        return (
            STORAGE_DIR
            / f"cockamamie_gobbledegook/embeddings/text_id_to_emb_{self.embeddings_type}.p"
        )

    @property
    def annotation_columns(self) -> List[str]:
        return ["is_funny"]

    @property
    def class_dims(self):
        return [2]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / "cockamamie_gobbledegook"

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
