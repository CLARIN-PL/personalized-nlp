import os
import pickle
from typing import Optional, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.tokenizer import get_text_data
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class CockamamieGobbledegookDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'cockamamie_gobbledegook/texts/',
            batch_size: int = 3000,
            embeddings_type: str = 'bert',
            language: str = 'english',
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            **kwargs,
    ):
        super().__init__()

        self.folds_num = 10
        self.data_dir = data_dir
        self.data_path = self.data_dir / 'cockamamie_gobbledegook.csv'
        self.data_url = None
        self.batch_size = batch_size
        self.split_sizes = split_sizes
        self.language = language
        self.embeddings_type = embeddings_type
        self.annotation_column = ['is_funny']
        self.text_column = 'text'
        if self.language != 'polish':
            self.text_column = f'text_{self.language}'

        self.word_stats_annotation_column = 'is_funny'
        self.embeddings_path = STORAGE_DIR / \
                               f'cockamamie_gobbledegook/embeddings/text_id_to_emb_{embeddings_type}_{language}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.normalize = normalize

    @property
    def class_dims(self):
        return [2]

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'cockamamie_texts_only_controversial_a.csv')
        self.data.loc[:, 'text'] = self.data.loc[:, 'text_' + self.language]
        #self.data.dropna(inplace=True)

        self.annotations = pd.read_csv(
            self.data_dir / 'cockamamie_annotations_only_controversial_a.csv').dropna()
        # self.annotators = pd.read_csv(
        #     self.data_dir / 'cockamamie_gobbledegook_annotators.csv')

        if not os.path.exists(self.embeddings_path):
            self._create_embeddings()

        if self.normalize:
            self.normalize_labels()

        text_idx_to_emb = pickle.load(open(self.embeddings_path, 'rb'))
        embeddings = []
        for text_idx in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_idx])

        self.text_embeddings = torch.tensor(embeddings)

    def normalize_labels(self):
        annotation_column = self.annotation_column
        df = self.annotations

        mins = df.loc[:, annotation_column].values.min(axis=0)
        df.loc[:, annotation_column] = (df.loc[:, annotation_column] - mins)

        maxes = df.loc[:, annotation_column].values.max(axis=0)
        df.loc[:, annotation_column] = df.loc[:, annotation_column] / maxes

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
            personal_df, self.annotation_column)

        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how='left')

        self.annotator_biases = annotator_biases.set_index(
            'annotator_id').sort_index().fillna(0)

    def _assign_splits(self):
        sizes = [0.55, 0.15, 0.15, 0.15]
        self.data = split_texts(self.data, sizes)

    def setup(self, stage: Optional[str] = None) -> None:
        data = self.data
        annotations = self.annotations

        self.text_id_idx_dict = data.loc[:, ['text_id']].reset_index(
        ).set_index('text_id').to_dict()['index']

        annotator_id_category = annotations['annotator_id'].astype('category')
        self.annotator_id_idx_dict = {a_id: idx for idx, a_id in enumerate(annotator_id_category.cat.categories)}

        self._assign_folds()
        self._assign_splits()

        self.compute_word_stats()

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def train_dataloader(self, test_fold=None) -> DataLoader:
        annotations = self.annotations
        data = self.data

        if test_fold is not None:
            val_fold = (test_fold + 1) % self.folds_num
            # all annotations from train folds
            annotations = annotations.loc[~annotations.fold.isin([test_fold, val_fold])]

            # past annotations for train and validation folds
            personal_df = self.annotations[self.annotations.text_id.isin(data[data.split == 'past'].text_id.values)]
            personal_df = personal_df[personal_df.fold.isin([test_fold, val_fold])]

            annotations = pd.concat([annotations, personal_df])

        train_X, train_y = self._get_data_by_split(annotations, self.train_split_names)
        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        train_dataset = BatchIndexedDataset(
            train_X, train_y, text_features=text_features, annotator_features=annotator_features)

        return self._prepare_dataloader(train_dataset)
