import os
import zipfile
import pickle
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import urllib
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from personalized_nlp.settings import PROJECT_DIR, STORAGE_DIR
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.tokenizer import get_text_data
from personalized_nlp.utils.embeddings import create_embeddings
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class WikiDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'wiki_data',
            batch_size: int = 3000,
            embeddings_type: str = 'bert',
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_path = ''
        self.data_url = None

        self.batch_size = batch_size

        self.annotation_column = ''
        self.word_stats_annotation_column = ''
        self.embeddings_path = ''
        self.embeddings_type = embeddings_type

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

        self.folds_num = 10

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

        self.annotations = pd.read_csv(
            self.data_dir / (self.annotation_column + '_annotations.tsv'), sep='\t')
        self.annotations = self._remap_column_names(self.annotations)

        self.annotators = pd.read_csv(
            self.data_dir / (self.annotation_column + '_worker_demographics.tsv'), sep='\t')
        self.annotators = self._remap_column_names(self.annotators)

        if not os.path.exists(self.embeddings_path):
            self._create_embeddings()

        text_idx_to_emb = pickle.load(open(self.embeddings_path, 'rb'))
        embeddings = []
        for text_id in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_id])

        assert len(self.data.index) == len(embeddings)

        self.text_embeddings = torch.tensor(embeddings)

    def compute_word_stats(
        self,
        min_word_count=100,
        min_std=0.0,
        words_per_text=100
    ):

        word_stats_annotation_column = self.word_stats_annotation_column or self.annotation_column
        _, self.text_tokenized, self.idx_to_word, self.tokens_sorted, self.word_stats = get_text_data(self.data,
                                                                                                      self.annotations,
                                                                                                      min_word_count=min_word_count,
                                                                                                      min_std=min_std,
                                                                                                      words_per_text=words_per_text,
                                                                                                      annotation_column=word_stats_annotation_column)

    def compute_annotator_biases(self, personal_df: pd.DataFrame):
        annotator_id_df = pd.DataFrame(
            self.annotations.annotator_id.unique(), columns=['annotator_id'])

        annotator_biases = get_annotator_biases(
            personal_df, [self.annotation_column])
        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how='left')
        self.annotator_biases = annotator_biases.set_index(
            'annotator_id').sort_index().fillna(0)

    def setup(self, stage: Optional[str] = None) -> None:
        data = self.data
        annotations = self.annotations

        self.text_id_idx_dict = data.loc[:, ['text_id']].reset_index(
        ).set_index('text_id').to_dict()['index']

        annotator_id_category = annotations['annotator_id'].astype('category')
        self.annotator_id_idx_dict = {a_id: idx for idx, a_id in enumerate(
            annotator_id_category.cat.categories)}

        self._assign_folds()

        self.compute_word_stats()

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'dev']
        self.compute_annotator_biases(personal_df)

    def get_conformity(self, annotations: pd.DataFrame = None) -> pd.DataFrame:
        if annotations is None:
            annotations = self.annotations

        df = annotations.copy()
        column = self.annotation_column

        mean_score = df.groupby('text_id').agg(score_mean=(column, 'mean'))
        df = df.merge(mean_score.reset_index())

        df['text_major_vote'] = (df['score_mean'] > 0.5).astype(int)

        df['is_major_vote'] = df['text_major_vote'] == df[column]
        df['is_major_vote'] = df['is_major_vote'].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        conformity_df = df.groupby('annotator_id').agg(
            conformity=('is_major_vote', 'mean'))
        conformity_df['pos_conformity'] = positive_df.groupby(
            'annotator_id').agg(pos_conformity=('is_major_vote', 'mean'))
        conformity_df['neg_conformity'] = negative_df.groupby(
            'annotator_id').agg(neg_conformity=('is_major_vote', 'mean'))

        return conformity_df
