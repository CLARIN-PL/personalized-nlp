from pathlib import Path
from typing import Tuple

import pandas as pd

from personalized_active_learning.datamodules.base import BaseDataModule, SplitMode
from settings import DATA_DIR


class WikiDataModule(BaseDataModule):
    """Base datamodule for wiki dataset.

    Id defines the processing of wiki data.
    The subclasses should only precise which problem we are trying to solve
    (e.g. toxicity).

    """

    @property
    def stratification_type(self) -> str:
        # TODO: Leftover of previous code
        if self.split_mode == SplitMode.TEXTS:
            stratification_type = "texts"
        elif self.split_mode == SplitMode.USERS:
            stratification_type = "users"
        else:
            raise Exception(f"Unsupported split mode {self.split_mode.value}")
        return stratification_type

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "wiki_data"

    @property
    def worker_demographics_file(self) -> str:
        return f"{self.annotation_columns[0]}_worker_demographics.tsv"

    @property
    def annotations_file(self) -> str:
        return f"{self.annotation_columns[0]}_annotations_{self.stratification_type}_folds.csv"  # noqa: E501

    @property
    def data_file(self) -> str:
        return f"{self.annotation_columns[0]}_annotated_comments_processed.csv"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _remap_column_names(self, df):
        mapping = {"rev_id": "text_id", "worker_id": "annotator_id", "comment": "text"}
        df = df.rename(columns=mapping)
        return df

    def load_data_and_annotations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv(self.data_dir / self.data_file)
        data = self._remap_column_names(data)
        data["text"] = data["text"].str.replace("NEWLINE_TOKEN", "  ")

        annotations = pd.read_csv(self.data_dir / self.annotations_file)
        annotations = self._remap_column_names(annotations)
        return data, annotations
