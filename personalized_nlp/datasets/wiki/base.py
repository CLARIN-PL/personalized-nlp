import os
import zipfile
from typing import List

import pandas as pd
import urllib

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class WikiDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'wiki_data',
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.data_path = ''
        self.data_url = None

        self.annotation_column = ''
        self.word_stats_annotation_column = ''
        self.embeddings_path = ''

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def texts_clean(self):
        texts = self.data.text.to_list()
        texts = [c.replace('NEWLINE_TOKEN', ' ') for c in texts]

        return texts

    def download_data(self) -> None:
        file_path = self.data_dir / 'temp.zip'
        urllib.request.urlretrieve(self.data_url, file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

    def _remap_column_names(self, df):
        mapping = {'rev_id': 'text_id',
                   'worker_id': 'annotator_id', 'comment': 'text'}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_path):
            self.download_data()

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

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'dev']
        self.compute_annotator_biases(personal_df)
