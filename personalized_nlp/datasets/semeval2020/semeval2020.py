from typing import List

import pandas as pd

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class SemevalDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'semeval2020/',
            language: str = 'english',
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.data_path = self.data_dir / 'texts.csv'
        self.split_sizes = split_sizes
        self.language = language
        self.annotation_column = ['grade']
        self.original_column = 'original'
        self.edited_column = 'edited'
        # if self.language != 'polish':
        #     self.text_column = f'text_{self.language}'

        self.word_stats_annotation_column = 'grade'
        self.embeddings_path = STORAGE_DIR / \
            f'semeval2020/embeddings/text_id_to_emb_{self.embeddings_type}_{language}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.normalize = normalize

    @property
    def class_dims(self):
        return [4] * 1

    @property
    def texts_clean(self):
        return self.data[self.original_column].to_list(), self.data[self.edited_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'texts.csv')
        self.data.loc[:, 'text_1'] = self.data.loc[:, 'original']
        self.data.loc[:, 'text_2'] = self.data.loc[:, 'edited']

        self.annotations = pd.read_csv(
            self.data_dir / 'annotations.csv').dropna()
        self.annotators = pd.read_csv(
            self.data_dir / 'annotators.csv')

        if self.normalize:
            self.normalize_labels()

        self._assign_splits()

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def normalize_labels(self):
        annotation_column = self.annotation_column
        df = self.annotations

        mins = df.loc[:, annotation_column].values.min(axis=0)
        df.loc[:, annotation_column] = (df.loc[:, annotation_column] - mins)

        maxes = df.loc[:, annotation_column].values.max(axis=0)
        df.loc[:, annotation_column] = df.loc[:, annotation_column] / maxes

    def _assign_splits(self):
        sizes = [0.55, 0.15, 0.15, 0.15]
        self.data = split_texts(self.data, sizes)
