from re import sub
from typing import List

import pandas as pd
import os
from pathlib import Path

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class MergedBinarizedHumorDatasetsDataModule(BaseDataModule):

    @property
    def annotations_file(self) -> str:
        return f"annotations.csv"

    @property
    def data_file(self) -> str:
        return "data.csv"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "merged_binarized_humor_datasets"

    @property
    def annotation_columns(self) -> List[str]:
        return ["is_funny_binarized"]

    @property
    def class_dims(self):
        return [2]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / self.data_file).dropna()

        self.data = self.data.replace('\n', ' ', regex=True)

        self.annotations = pd.read_csv(self.data_dir / self.annotations_file)

        self.annotations = self.annotations.rename(
            columns={'user_id': 'annotator_id'})

        # self.annotations = self.annotations[self.annotations['text_id'].isin(
            # self.data['text_id'].unique().tolist())]
        # self.annotations = self.annotations.drop_duplicates(subset=["text_id"])

        print(
            f'self.data size: {len(self.data)}',
            f'self.annotations size: {len(self.annotations)}',
            f'text_id diff: {set(self.annotations.text_id.unique().tolist()) - set(self.data.text_id.unique().tolist())}',
            sep='\n')
