import torch
import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.embeddings import create_embeddings


class BaseDataModule(LightningDataModule):

    @property
    def words_number(self):
        return self.tokens_sorted.max() + 1

    @property
    def annotators_number(self):
        return max(self.annotator_id_idx_dict.values()) + 2

    @property
    def text_embedding_dim(self):
        if self.embeddings_type in ['xlmr', 'bert']:
            return 768
        else:
            return 1024

    @property
    def annotations_with_data(self):
        return self.annotations.merge(self.data)

    def _create_embeddings(self):
        texts = self.texts_clean
        embeddings_path = self.embeddings_path

        if self.embeddings_type == 'xlmr':
            model_name = 'xlm-roberta-base'
        elif self.embeddings_type == 'bert':
            model_name = 'bert-base-cased'
        elif self.embeddings_type == 't5':
            model_name = 'google/t5-large-ssm'
        elif self.embeddings_type == 'deberta':
            model_name = 'microsoft/deberta-large'
        elif self.embeddings_type == 'labse':
            model_name = 'sentence-transformers/LaBSE'

        use_cuda = torch.cuda.is_available()
        create_embeddings(texts, embeddings_path,
                          model_name=model_name, use_cuda=use_cuda)

    def _prepare_dataloader(self, dataset, shuffle=True):
        if shuffle:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False)
        else:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False)

        return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None)

    def train_dataloader(self, test_fold=None) -> DataLoader:
        annotations = self.annotations

        if test_fold is not None:
            val_fold = (test_fold + 1) % self.folds_num
            annotations = annotations[~annotations.fold.isin(
                [test_fold, val_fold])]

        train_X, train_y = self._get_data_by_split(
            annotations, self.train_split_names)
        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        train_dataset = BatchIndexedDataset(
            train_X, train_y, text_features=text_features, annotator_features=annotator_features)

        return self._prepare_dataloader(train_dataset)

    def val_dataloader(self, test_fold=None) -> DataLoader:
        annotations = self.annotations

        if test_fold is not None:
            val_fold = (test_fold + 1) % self.folds_num
            annotations = annotations[annotations.fold.isin([val_fold])]

        dev_X, dev_y = self._get_data_by_split(
            annotations, self.val_split_names)

        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        dev_dataset = BatchIndexedDataset(
            dev_X, dev_y, text_features=text_features, annotator_features=annotator_features)

        return self._prepare_dataloader(dev_dataset, shuffle=False)

    def test_dataloader(self, test_fold=None) -> DataLoader:
        annotations = self.annotations

        if test_fold is not None:
            annotations = annotations[annotations.fold.isin([test_fold])]

        test_X, test_y = self._get_data_by_split(
            annotations, self.test_split_names)
        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        test_dataset = BatchIndexedDataset(
            test_X, test_y, text_features=text_features, annotator_features=annotator_features)

        return self._prepare_dataloader(test_dataset, shuffle=False)

    def _get_text_features(self):
        return {
            'embeddings': self.text_embeddings,
            'text_tokenized': self.text_tokenized,
            'tokens_sorted': self.tokens_sorted,
            'raw_texts': self.data['text'].values
        }

    def _get_annotator_features(self):
        return {
            'annotator_biases': self.annotator_biases.values.astype(float)
        }

    def _assign_folds(self):
        annotations = self.annotations
        annotator_ids = annotations['annotator_id'].unique()
        np.random.shuffle(annotator_ids)

        folded_workers = np.array_split(annotator_ids, self.folds_num)

        annotations['fold'] = 0
        for i in range(self.folds_num):
            annotations.loc[annotations.annotator_id.isin(
                folded_workers[i]), 'fold'] = i

    def _get_data_by_split(self, annotations: pd.DataFrame, splits: str):
        data = self.data

        df = annotations.loc[annotations.text_id.isin(
            data[data.split.isin(splits)].text_id.values)]
        X = df.loc[:, ['text_id', 'annotator_id']]
        y = df[self.annotation_column]

        X['text_id'] = X['text_id'].apply(
            lambda r_id: self.text_id_idx_dict[r_id])
        X['annotator_id'] = X['annotator_id'].apply(
            lambda w_id: self.annotator_id_idx_dict[w_id])

        X, y = X.values, y.values

        if y.ndim < 2:
            y = y[:, None]

        return X, y

    def get_X_y(self):
        annotations = self.annotations.copy()
        annotations = annotations.merge(self.data)
        embeddings = self.text_embeddings.to('cpu').numpy()

        annotations['text_idx'] = annotations['text_id'].apply(
            lambda r_id: self.text_id_idx_dict[r_id])
        annotations['annotator_idx'] = annotations['annotator_id'].apply(
            lambda w_id: self.annotator_id_idx_dict[w_id])

        X = np.vstack([embeddings[i] for i in annotations['text_idx'].values])
        y = annotations[self.annotation_column].values

        return X, y
