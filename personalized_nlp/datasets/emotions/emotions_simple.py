from typing import List

import pandas as pd
import os

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class EmotionsSimpleDataModule(BaseDataModule):
    def __init__(
            self,
            language: str = 'english',
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / 'emotions_simple_data'
        self.split_sizes = split_sizes
        self.language = language
        self.annotation_column = ['OCZEKIWANIE',
                                  'POBUDZENIE EMOCJONALNE',
                                  'RADOŚĆ',
                                  'SMUTEK',
                                  'STRACH',
                                  'WSTRĘT',
                                  'ZASKOCZENIE',
                                  'ZAUFANIE',
                                  'ZNAK EMOCJI',
                                  'ZŁOŚĆ']
        self.text_column = 'text'
        if self.language != 'polish':
            self.text_column = f'text_{self.language}'

        self.word_stats_annotation_column = 'POBUDZENIE EMOCJONALNE'
        self.embeddings_path = STORAGE_DIR / \
            f'emotions_simple_data/embeddings/text_id_to_emb_{self.embeddings_type}_{language}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.normalize = normalize

        os.makedirs(self.data_dir / 'embeddings', exist_ok=True)

    @property
    def class_dims(self):
        return [5] * 8 + [7, 5]

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'texts' / 'cawi2_texts_multilang.csv')
        self.data.loc[:, 'text'] = self.data.loc[:, 'text_' + self.language]

        self.annotations = pd.read_csv(
            self.data_dir / 'texts' / 'cawi2_annotations.csv').dropna()
        self.annotators = pd.read_csv(
            self.data_dir / 'texts' / 'cawi2_annotators.csv')

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
        self.data = split_texts(self.data, self.split_sizes)