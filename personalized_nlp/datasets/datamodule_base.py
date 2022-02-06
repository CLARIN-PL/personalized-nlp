import torch
import numpy as np
import pandas as pd
import os
import pickle

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.embeddings import create_embeddings
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.controversy import get_texts_entropy, get_texts_std
from personalized_nlp.utils.tokenizer import get_text_data
from personalized_nlp.settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS

from typing import Any, Dict, List, Tuple, Optional


class BaseDataModule(LightningDataModule):
    @property
    def class_dims(self):
        raise NotImplementedError()

    @property
    def words_number(self) -> int:
        return self.tokens_sorted.max() + 1

    @property
    def annotators_number(self) -> int:
        return max(self.annotator_id_idx_dict.values()) + 2

    @property
    def text_embedding_dim(self) -> int:
        if not self.embeddings_type in EMBEDDINGS_SIZES:
            raise NotImplementedError()

        return EMBEDDINGS_SIZES[self.embeddings_type]

    @property
    def annotations_with_data(self) -> pd.DataFrame:
        return self.annotations.merge(self.data)

    def __init__(
        self,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dims=None,
        batch_size: int = 3000,
        embeddings_type: str = "bert",
        major_voting: bool = False,
        folds_num: int = 10,
        past_annotations_limit: int = None,
        **kwargs
    ):

        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=dims,
        )

        self.major_voting = major_voting
        self.batch_size = batch_size
        self.embeddings_type = embeddings_type
        self.folds_num = folds_num
        self.past_annotations_limit = past_annotations_limit
        self.annotation_column: list[str] = []

    def _create_embeddings(self, use_cuda: Optional[bool] = None) -> None:
        texts = self.texts_clean
        embeddings_path = self.embeddings_path

        if self.embeddings_type in TRANSFORMER_MODEL_STRINGS:
            model_name = TRANSFORMER_MODEL_STRINGS[self.embeddings_type]
        else:
            model_name = self.embeddings_type

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        create_embeddings(
            texts, embeddings_path, model_name=model_name, use_cuda=use_cuda
        )

    def compute_word_stats(
        self, min_word_count: int = 100, min_std: float = 0.0, words_per_text: int = 100
    ):
        word_stats_annotation_column = (
            self.word_stats_annotation_column or self.annotation_column
        )

        annotations = self.annotations
        data = self.data
        train_split_names = self.train_split_names

        # select train annotations for word stat computations
        annotations = annotations.loc[
            annotations.text_id.isin(
                data[data.split.isin(train_split_names)].text_id.values
            )
        ].copy()

        (
            _,
            self.text_tokenized,
            self.idx_to_word,
            self.tokens_sorted,
            self.word_stats,
        ) = get_text_data(
            self.data,
            annotations,
            min_word_count=min_word_count,
            min_std=min_std,
            words_per_text=words_per_text,
            annotation_column=word_stats_annotation_column,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        data = self.data
        annotations = self.annotations

        if self.major_voting:
            self.compute_major_votes()

        if not os.path.exists(self.embeddings_path):
            self._create_embeddings()

        text_idx_to_emb = pickle.load(open(self.embeddings_path, "rb"))
        embeddings = []
        for text_id in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_id])

        assert len(self.data.index) == len(embeddings)

        self.text_embeddings = torch.tensor(embeddings)

        self.text_id_idx_dict = (
            data.loc[:, ["text_id"]]
            .reset_index()
            .set_index("text_id")
            .to_dict()["index"]
        )

        annotator_id_category = annotations["annotator_id"].astype("category")
        self.annotator_id_idx_dict = {
            a_id: idx for idx, a_id in enumerate(annotator_id_category.cat.categories)
        }

        if self.past_annotations_limit is not None:
            self.limit_past_annotations(self.past_annotations_limit)

        self._assign_folds()
        self.compute_word_stats()

    def compute_major_votes(self) -> None:
        """Computes mean votes for every texts and replaces
        each annotator with dummy annotator with id = 0"""
        annotations = self.annotations

        annotations["annotator_id"] = 0
        major_votes = annotations.groupby("text_id")[self.annotation_column].mean()
        major_votes = major_votes.round()

        self.annotations = major_votes.reset_index()
        self.annotations["annotator_id"] = 0

    def compute_annotator_biases(self, personal_df: pd.DataFrame):
        if self.past_annotations_limit is not None:
            self.limit_past_annotations(self.past_annotations_limit)
            
        annotator_id_df = pd.DataFrame(
            self.annotations.annotator_id.unique(), columns=["annotator_id"]
        )

        annotation_columns = self.annotation_column
        if isinstance(self.annotation_column, str):
            annotation_columns = [annotation_columns]

        annotator_biases = get_annotator_biases(personal_df, annotation_columns)
        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how="left"
        )
        self.annotator_biases = (
            annotator_biases.set_index("annotator_id").sort_index().fillna(0)
        )

    def train_dataloader(
        self, test_fold: int = None, shuffle: bool = True
    ) -> DataLoader:
        """Returns dataloader for train part of the dataset.

        :param test_fold: Number of test fold used in test, defaults to None
        :type test_fold: int, optional
        :param shuffle: if true, shuffles data during training, defaults to True
        :type shuffle: bool, optional
        :return: train dataloader for the dataset
        :rtype: DataLoader
        """
        annotations = self.annotations
        data = self.data

        if test_fold is not None:
            val_fold = (test_fold + 1) % self.folds_num
            # all annotations from train folds
            annotations = annotations.loc[~annotations.fold.isin([test_fold, val_fold])]

            # past annotations for test and validation folds
            personal_df = self.annotations[
                self.annotations.text_id.isin(data[data.split == "past"].text_id.values)
            ]
            personal_df = personal_df[personal_df.fold.isin([test_fold, val_fold])]

            annotations = pd.concat([annotations, personal_df])

        train_X, train_y = self._get_data_by_split(annotations, self.train_split_names)
        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        train_dataset = BatchIndexedDataset(
            train_X,
            train_y,
            text_features=text_features,
            annotator_features=annotator_features,
        )

        return self._prepare_dataloader(train_dataset, shuffle=shuffle)

    def val_dataloader(self, test_fold=None) -> DataLoader:
        """Returns dataloader for validation part of the dataset.

        :param test_fold: number of test fold used in test, defaults to None.
        Number of validation fold is calculated as: (test_fold + 1) % folds_num
        :type test_fold: int, optional
        :return: validation dataloader for the dataset
        :rtype: DataLoader
        """
        annotations = self.annotations

        if test_fold is not None:
            val_fold = (test_fold + 1) % self.folds_num
            annotations = annotations[annotations.fold.isin([val_fold])]

        dev_X, dev_y = self._get_data_by_split(annotations, self.val_split_names)

        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        dev_dataset = BatchIndexedDataset(
            dev_X,
            dev_y,
            text_features=text_features,
            annotator_features=annotator_features,
        )

        return self._prepare_dataloader(dev_dataset, shuffle=False)

    def test_dataloader(self, test_fold=None) -> DataLoader:
        """Returns dataloader for test part of the dataset.

        :param test_fold: Number of test fold used in test, defaults to None
        :type test_fold: int, optional
        :return: validation dataloader for the dataset
        :rtype: DataLoader
        """
        annotations = self.annotations

        if test_fold is not None:
            annotations = annotations[annotations.fold.isin([test_fold])]

        test_X, test_y = self._get_data_by_split(annotations, self.test_split_names)
        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        test_dataset = BatchIndexedDataset(
            test_X,
            test_y,
            text_features=text_features,
            annotator_features=annotator_features,
        )

        return self._prepare_dataloader(test_dataset, shuffle=False)

    def _prepare_dataloader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = True
    ):
        if shuffle:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False,
            )
        else:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False,
            )

        return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None)

    def _get_text_features(self) -> Dict[str, Any]:
        """Returns dictionary of features of all texts in the dataset.
        Each feature should be a numpy array of whatever dtype, with length
        equal to number of texts in the dataset. Features can be used by
        models during training.


        :return: dictionary of text features
        :rtype: Dict[str, Any]
        """
        return {
            "embeddings": self.text_embeddings,
            "text_tokenized": self.text_tokenized,
            "tokens_sorted": self.tokens_sorted,
            "raw_texts": self.data["text"].values,
        }

    def _get_annotator_features(self):
        """Returns dictionary of features of all annotators in the dataset.
        Each feature should be a numpy array of whatever dtype, with length
        equal to number of annotators in the dataset. Features can be used by
        models during training.


        :return: dictionary of annotator features
        :rtype: Dict[str, Any]
        """
        return {"annotator_biases": self.annotator_biases.values.astype(float)}

    def _assign_folds(self):
        """Randomly assign fold to each annotation."""
        annotations = self.annotations
        annotator_ids = annotations["annotator_id"].unique()
        np.random.shuffle(annotator_ids)

        folded_workers = np.array_split(annotator_ids, self.folds_num)

        annotations["fold"] = 0
        for i in range(self.folds_num):
            annotations.loc[
                annotations.annotator_id.isin(folded_workers[i]), "fold"
            ] = i

    def _get_data_by_split(
        self, annotations: pd.DataFrame, splits: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns annotations (coded indices of annotators and texts), and
        their labels in the dataset for given splits. Used during training.

        :param annotations: DataFrame of annotations from which the data will
        be extracted.
        :type annotations: pd.DataFrame
        :param splits: List of names of splits to be extracted
        :type splits: List[str]
        :return: tuple (X, y), where X is numpy array of (annotator_idx, text_idx)
        tuples and y is numpy array of labels for the annotations.
        :rtype: [type]
        """
        data = self.data

        df = annotations.loc[
            annotations.text_id.isin(data[data.split.isin(splits)].text_id.values)
        ]
        X = df.loc[:, ["text_id", "annotator_id"]]
        y = df[self.annotation_column]

        X["text_id"] = X["text_id"].apply(lambda r_id: self.text_id_idx_dict[r_id])
        X["annotator_id"] = X["annotator_id"].apply(
            lambda w_id: self.annotator_id_idx_dict[w_id]
        )

        X, y = X.values, y.values

        if y.ndim < 2:
            y = y[:, None]

        return X, y

    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns full dataset as X, y, where X is numpy array of embeddings,
        and y is numpy array of the labels. Can be easily used in some baseline
        models like sklearn logistic regression.

        :return: Tuple (X, y), where X is numpy array of embeddings,
        and y is numpy array of the labels
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        annotations = self.annotations.copy()
        annotations = annotations.merge(self.data)
        embeddings = self.text_embeddings.to("cpu").numpy()

        annotations["text_idx"] = annotations["text_id"].apply(
            lambda r_id: self.text_id_idx_dict[r_id]
        )
        annotations["annotator_idx"] = annotations["annotator_id"].apply(
            lambda w_id: self.annotator_id_idx_dict[w_id]
        )

        X = np.vstack([embeddings[i] for i in annotations["text_idx"].values])
        y = annotations[self.annotation_column].values

        return X, y

    def get_conformity(self, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        if annotations is None:
            annotations = self.annotations

        df = annotations.copy()
        column = self.annotation_column

        mean_score = df.groupby("text_id").agg(score_mean=(column, "mean"))
        df = df.merge(mean_score.reset_index())

        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)

        df["is_major_vote"] = df["text_major_vote"] == df[column]
        df["is_major_vote"] = df["is_major_vote"].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        conformity_df = df.groupby("annotator_id").agg(
            conformity=("is_major_vote", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_conformity=("is_major_vote", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_conformity=("is_major_vote", "mean")
        )

        return conformity_df

    def limit_past_annotations(self, limit: int):

        past_annotations = self.annotations.merge(self.data[self.data.split == "past"])

        text_stds = (
            past_annotations.groupby("text_id")[self.annotation_column]
            .agg("std")
            .reset_index()
        )
        text_stds.columns = ["text_id", "std"]

        past_annotations = past_annotations.merge(text_stds)

        controversial_annotations = past_annotations.groupby("annotator_id").apply(
            lambda x: x.sort_values(by="std", ascending=False)[:limit]
        )

        past_split_text_ids = self.data[self.data.split == "past"].text_id.tolist()
        non_past_annotations = self.annotations[
            ~self.annotations["text_id"].isin(past_split_text_ids)
        ]

        self.annotations = pd.concat([non_past_annotations, controversial_annotations])

    def compute_texts_controversy(self, mode: str = 'entropy', mean=False):
        if mode == 'entropy':
            return get_texts_entropy(self.annotations, self.annotation_column, mean=mean)
        else:
            return get_texts_std(self.annotations, self.annotation_column, mean=mean)

