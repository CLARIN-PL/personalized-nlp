import os
import pickle
from typing import Optional, List

import pandas as pd

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.tokenizer import get_text_data
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class CockamamieGobbledegookDataModule(BaseDataModule):
    def __init__(
            self,
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / 'cockamamie_gobbledegook'
        self.split_sizes = split_sizes
        self.annotation_column = ['is_funny']
        self.text_column = 'text'

        self.word_stats_annotation_column = 'is_funny'
        self.embeddings_path = STORAGE_DIR / \
                               f'cockamamie_gobbledegook/embeddings/text_id_to_emb_{self.embeddings_type}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.normalize = normalize

        os.makedirs(self.data_dir / 'embeddings', exist_ok=True)

    @property
    def class_dims(self):
        return [2]

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'texts' / 'cockamamie_gobbledegook_texts_e.csv')
        self.data['text'] = self.data['text_english']
        
        self.annotations = pd.read_csv(
            self.data_dir / 'texts' / 'cockamamie_annotations_only_controversial_a_non_empty.csv').dropna()

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