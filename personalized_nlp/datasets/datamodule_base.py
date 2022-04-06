import abc
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.utils.embeddings import create_embeddings
from personalized_nlp.utils.tokenizer import get_text_data
from personalized_nlp.utils.controversy import get_conformity
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule, abc.ABC):
    @abc.abstractproperty
    def class_dims(self) -> List[int]:
        raise NotImplementedError()

    @abc.abstractproperty
    def annotation_columns(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractproperty
    def embeddings_path(self) -> Path:
        raise NotImplementedError()

    @property
    def word_stats_annotation_column(self) -> str:
        return self.annotation_columns[0]

    @property
    def words_number(self) -> int:
        return self.tokens_sorted.max()

    @property
    def annotators_number(self) -> int:
        return max(self.annotator_id_idx_dict.values())

    @property
    def train_split_names(self) -> List[str]:
        if self.stratify_folds_by == "text":
            return ["none"]

        return ["present", "past"]

    @property
    def val_split_names(self) -> List[str]:
        if self.stratify_folds_by == "text":
            return ["none"]

        return ["future1"]

    @property
    def test_split_names(self) -> List[str]:
        if self.stratify_folds_by == "text":
            return ["none"]

        return ["future2"]

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
        normalize: bool = False,
        past_annotations_limit: Optional[int] = None,
        stratify_folds_by: Optional[str] = "users",
        split_sizes: Optional[List[str]] = None,
        **kwargs
    ):
        """_summary_

        Args:
            batch_size (int, optional): Batch size for data loaders. Defaults to 3000.
            embeddings_type (str, optional): string identifier of embedding. Defaults to "bert".
            major_voting (bool, optional): if true, use major voting. Defaults to False.
            folds_num (int, optional): Number of folds. Defaults to 10.
            normalize (bool, optional): Normalize labels to [0, 1] range with min-max method. Defaults to False.
            past_annotations_limit (Optional[int], optional): Maximum number of annotations in past dataset part. Defaults to None.
            ignore_train_test_val (bool, optional): If true, use only folds. Defaults to False.
            stratify_folds_by (str, optional): How to stratify annotations: 'texts' or 'users'. Defaults to 'texts'.
        """

        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=dims,
        )

        self.normalize = normalize
        self.major_voting = major_voting
        self.batch_size = batch_size
        self.embeddings_type = embeddings_type
        self.folds_num = folds_num
        self.past_annotations_limit = past_annotations_limit
        self.stratify_folds_by = stratify_folds_by

        self.split_sizes = (
            split_sizes if split_sizes is not None else [0.55, 0.15, 0.15, 0.15]
        )

        self.annotations = pd.DataFrame([])
        self.data = pd.DataFrame([])

        self._current_stats_fold: Optional[int] = None

    def _create_embeddings(self, use_cuda: Optional[bool] = None) -> None:
        texts = self.data["text"].tolist()
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
        self,
        fold_nums: Optional[List[int]] = None,
        min_word_count: int = 100,
        min_std: float = 0.0,
        words_per_text: int = 100,
    ):
        word_stats_annotation_column = self.word_stats_annotation_column

        annotations = self.annotations
        data = self.data

        train_split_names = self.train_split_names

        if self.stratify_folds_by != "texts":
            train_split_mask = annotations.text_id.isin(
                data[data.split.isin(train_split_names)].text_id.values
            )
            annotations = annotations.loc[train_split_mask]

        if fold_nums is not None:
            annotations = annotations.loc[annotations.fold.isin(fold_nums)].copy()

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

        if self.stratify_folds_by != "texts":
            """Fixed text split - we can precompute annotator biases and
            limit past annotations.
            """
            self._assign_splits()
            personal_df = self.annotations_with_data.loc[
                self.annotations_with_data.split == "past"
            ]
            self.compute_annotator_biases(personal_df)

            if self.past_annotations_limit is not None:
                self.limit_past_annotations(self.past_annotations_limit)
        else:
            self.annotations["split"] = "none"

        if self.normalize:
            self._normalize_labels()

        self._assign_folds()

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

        self._original_annotations = annotations.copy()

        if self.major_voting:
            self.compute_major_votes(0)

    def _assign_splits(self):
        self.data = split_texts(self.data, self.split_sizes)

    def _normalize_labels(self):
        annotation_columns = self.annotation_columns
        df = self.annotations

        mins = df.loc[:, annotation_columns].values.min(axis=0)
        df.loc[:, annotation_columns] = df.loc[:, annotation_columns] - mins

        maxes = df.loc[:, annotation_columns].values.max(axis=0)
        df.loc[:, annotation_columns] = df.loc[:, annotation_columns] / maxes

        df.loc[:, annotation_columns] = df.loc[:, annotation_columns].fillna(0)

    def compute_major_votes(self, test_fold: int) -> None:
        """Computes mean votes for texts in train folds.
        Works only for 'texts' folds"""
        annotations = self._original_annotations.copy()
        annotation_columns = self.annotation_columns
        val_fold = (test_fold + 1) % self.folds_num

        text_id_to_fold = (
            annotations.loc[:, ["text_id", "fold"]]
            .drop_duplicates()
            .set_index("text_id")
            .to_dict()["fold"]
        )

        val_test_annotations = annotations.loc[
            annotations.fold.isin([val_fold, test_fold])
        ]
        annotations = annotations.loc[~annotations.fold.isin([val_fold, test_fold])]

        dfs = []
        for col in annotation_columns:
            if self.normalize:
                aggregate_lambda = lambda x: x.mean()
            else:
                aggregate_lambda = lambda x: pd.Series.mode(x)[0]

            dfs.append(annotations.groupby("text_id")[col].apply(aggregate_lambda))

        annotations = pd.concat(dfs, axis=1).reset_index()
        annotations["annotator_id"] = 0
        annotations["split"] = "none"
        annotations["fold"] = annotations["text_id"].map(text_id_to_fold)

        self.annotations = pd.concat([annotations, val_test_annotations])

    def compute_annotator_biases(self, personal_df: pd.DataFrame):
        if self.past_annotations_limit is not None:
            self.limit_past_annotations(self.past_annotations_limit)

        annotator_id_df = pd.DataFrame(
            self._original_annotations.annotator_id.unique(), columns=["annotator_id"]
        )

        annotation_columns = self.annotation_columns

        annotator_biases = get_annotator_biases(personal_df, annotation_columns)
        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how="left"
        )
        self.annotator_biases = (
            annotator_biases.set_index("annotator_id").sort_index().fillna(0)
        )

    def _recompute_stats_for_fold(self, test_fold):
        """Recalculate annotator biases and words stats for
        given text test fold"""
        if self.stratify_folds_by != "texts":
            return

        if self._current_stats_fold == test_fold:
            return

        val_fold = (test_fold + 1) % self.folds_num

        train_folds = list(range(self.folds_num))
        train_folds.remove(val_fold)
        train_folds.remove(test_fold)

        self.compute_word_stats(fold_nums=train_folds)

        if self.major_voting:
            self.compute_major_votes(test_fold)

        train_annotations = self.annotations
        train_annotations = train_annotations.loc[
            ~train_annotations.fold.isin([val_fold, test_fold])
        ]
        self.compute_annotator_biases(train_annotations)

        self._current_stats_fold = test_fold

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

        if test_fold is not None and self.stratify_folds_by is not None:
            val_fold = (test_fold + 1) % self.folds_num
            # all annotations from train folds
            annotations = annotations.loc[~annotations.fold.isin([test_fold, val_fold])]

            if self.stratify_folds_by == "users":
                # past annotations for test and validation folds
                personal_df = self.annotations[
                    self.annotations.text_id.isin(
                        data[data.split == "past"].text_id.values
                    )
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

        if test_fold is not None and self.stratify_folds_by is not None:
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

        if test_fold is not None and self.stratify_folds_by is not None:
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
            order_sampler_cls = torch.utils.data.sampler.RandomSampler
        else:
            order_sampler_cls = torch.utils.data.sampler.SequentialSampler

        sampler = torch.utils.data.sampler.BatchSampler(
            order_sampler_cls(dataset),
            batch_size=self.batch_size,
            drop_last=False,
        )

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
        )

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
        if self.stratify_folds_by is None:
            return

        if self.stratify_folds_by == "texts":
            stratify_column = "text_id"
        else:
            stratify_column = "annotator_id"

        annotations = self.annotations
        ids = annotations[stratify_column].unique()
        np.random.shuffle(ids)

        folded_ids = np.array_split(ids, self.folds_num)

        annotations["fold"] = 0
        for i in range(self.folds_num):
            annotations.loc[
                annotations[stratify_column].isin(folded_ids[i]), "fold"
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

        df = annotations
        if self.stratify_folds_by != "texts":
            df = annotations.loc[
                annotations.text_id.isin(data[data.split.isin(splits)].text_id.values)
            ]

        X = df.loc[:, ["text_id", "annotator_id"]]
        y = df[self.annotation_columns]

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
        y = annotations[self.annotation_columns].values

        return X, y

    def get_conformity(self, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        return get_conformity(self.annotations, self.annotation_columns)

    def limit_past_annotations(self, limit: int):
        past_annotations = self.annotations.merge(self.data[self.data.split == "past"])

        text_stds = (
            past_annotations.groupby("text_id")[self.annotation_columns]
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
