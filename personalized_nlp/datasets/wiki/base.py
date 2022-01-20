import os
import zipfile
from typing import List

import pandas as pd
import urllib

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class WikiDataModule(BaseDataModule):
    def __init__(
            self,
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / 'wiki_data'

        self.annotation_column = ''
        self.word_stats_annotation_column = ''
        self.embeddings_path = ''

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.split_sizes = split_sizes
    
        os.makedirs(self.data_dir / 'embeddings', exist_ok=True)
        
    @property
    def class_dims(self):
        return [2]

    @property
    def texts_clean(self):
        texts = self.data.text.to_list()
        texts = [c.replace('NEWLINE_TOKEN', ' ') for c in texts]

        return texts

    def _remap_column_names(self, df):
        mapping = {'rev_id': 'text_id',
                   'worker_id': 'annotator_id', 'comment': 'text'}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / (self.annotation_column + '_annotated_comments.tsv'), sep='\t')
        self.data = self._remap_column_names(self.data)
        self.data['text'] = self.data['text'].str.replace(
            'NEWLINE_TOKEN', '  ')

        self.annotators = pd.read_csv(
            self.data_dir / (self.annotation_column + '_worker_demographics.tsv'), sep='\t')
        self.annotators = self._remap_column_names(self.annotators)

        self.annotations = pd.read_csv(
            self.data_dir / (self.annotation_column + '_annotations.tsv'), sep='\t')
        self.annotations = self._remap_column_names(self.annotations)

        if self.limit_annotations_function is not None:
            self.annotations = self.limit_annotations_function(
                self.data,
                self.annotations,
                self.annotators
            )

        self._assign_splits()
        
        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def _assign_splits(self):
        self.data = split_texts(self.data, self.split_sizes)
