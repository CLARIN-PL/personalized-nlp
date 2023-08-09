import pandas as pd
import os

from pathlib import Path
from typing import List

from settings import DATA_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class MergedHumorDatasetsMajorityDataModule(BaseDataModule):

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "merged_humor_datasets" / "majority_voting_personalized_datasets"

    @property
    def data_file(self) -> str:
        return "data.csv"

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
