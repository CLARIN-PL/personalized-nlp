import abc
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader

from personalized_active_learning.datasets.types import TextFeaturesBatchDataset
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.utils.embeddings import create_embeddings
from personalized_nlp.utils.finetune import finetune_datamodule_embeddings
from settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS


class SplitMode(Enum):
    """Defines a way of splitting data into training, validation and test subsets.

    Available modes:
        TEXTS: Data are split based on texts, so training data contains texts
            that are not present in test or validation data.
        USERS: Data are split based on users, so training data contains texts
            annotated by different users then users annotating test and validation data.
        PREDEFINED: Data are split based on column `split` available in original data.

    """

    TEXTS = "TEXTS"
    USERS = "USERS"
    PREDEFINED = "PREDEFINED"


class BaseDataset(LightningDataModule, abc.ABC):
    """The base class from each dataset class should derive."""

    @property
    @abc.abstractmethod
    def classes_dimensions(self) -> List[int]:
        """Get the class dimensions.

        The length of this list must be the same as  `self.annotation_columns`.
        Example:
            `self.annotation_columns = [`sentiment`, `sarcasm`]`
            `self.classes_dimensions = [`3`, `2`]`

        Returns:
            List of class dimensions, each one corresponds to separated supervised task.
        """
        # TODO: Consider returning int instead (I.E. single task per datamodule).

    @property
    @abc.abstractmethod
    def annotation_columns(self) -> List[str]:
        """Get the annotation columns.

        Each collum corresponds to separated supervised task, e.g.:
        [`humor`, `sarcasm`]

        """
        # TODO: Consider returning string instead (I.E. single task per datamodule).

    @property
    @abc.abstractmethod
    def embeddings_path(self) -> Path:
        """Get the path to texts' embeddings.

        Returns:
            The embeddings path to texts' embeddings.

        """

    # TODO: Remove, will be used in load_data_and_annotations
    @property
    @abc.abstractmethod
    def annotations_file_relative_path(self) -> str:
        """Get the relative path to csv file containing annotations.

        `self.data_dir / self.annotations_file_relative_path` should point
        to the csv file containing annotations.

        Returns:
            the relative path to csv file containing annotations.

        """

    # TODO: Remove, will be used in load_data_and_annotations
    @property
    @abc.abstractmethod
    def data_file_relative_path(self) -> str:
        """Get the relative path to csv file containing data.

        `self.data_dir / self.data_file_relative_path` should point
        to the csv file containing data.

        Returns:
            the relative path to csv file containing data.

        """

    # TODO: Remove, will be used in load_data_and_annotations
    @property
    @abc.abstractmethod
    def data_dir(self) -> Path:
        """Get the path to directory containing data.

        Returns:
            The path to directory containing data.

        """

    @abc.abstractmethod
    def load_data_and_annotations(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data (texts) and annotations.

        Additional preprocessing & filtering should go here.

        Returns:
            Tuple of texts and annotations.

        """

    @property
    def annotators_number(self) -> int:
        """Get the number of annotators.

        Returns:
            The number of annotators.

        """
        return max(self.annotations["annotator_id"]) + 1

    @property
    def num_annotations(self) -> int:
        """Get the number of annotations.

        Returns:
            The number of annotations.

        """
        return len(self.annotations)

    @property
    def train_folds_indices(self) -> List[int]:
        """Get the list of training folds' indices.

        Returns:
            The list of training folds' indices.

        """
        all_folds = range(self.folds_num)
        exclude_folds = [self.val_fold_index, self.test_fold_index]

        return [fold for fold in all_folds if fold not in exclude_folds]

    @property
    def val_fold_index(self) -> int:
        """Get the index of validation fold.

        Returns:
            The index of validation fold.

        """
        return (self._test_fold_index + 1) % self.folds_num

    @property
    def test_fold_index(self) -> int:
        return self._test_fold_index

    # TODO: Rewrite embeddings
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
        batch_size: int = 3000,
        split_mode: SplitMode = SplitMode.TEXTS,
        embeddings_type: str = "labse",  # TODO: Change to class
        major_voting: bool = False,  # TODO: Not sure if we need that
        test_major_voting: bool = False,  # TODO: Not sure if we need that
        folds_num: int = 10,  # TODO: Can be obtained from data?
        past_annotations_limit: Optional[int] = None,  # TODO: Not sure if we need that
        split_sizes: Optional[
            List[str]
        ] = None,  # TODO: Used only when split mode is user, ugly
        use_finetuned_embeddings: bool = False,  # TODO: I would like to eliminate that
        test_fold_index: int = 0,
        min_annotations_per_user_in_fold: Optional[int] = None,
        seed: int = 22,  # TODO: It shouldn't be use by datamodule but higher!!!
        use_cuda: bool = False,  # TODO: Moved to class responsible for embeddings
    ):
        """Initialize object.

        Args:
            batch_size: The size of batch.
            split_mode: The split mode used to divide data into training, val & test.
            embeddings_type: Will be removed.
            major_voting: If `True` do not use personalization on training data.
                I.E. Text annotated by 5 annotators will be transformed to single text.
            test_major_voting: If `True` do not use personalization on val & test data.
                I.E. Text annotated by 5 annotators will be transformed to single text.
            folds_num: The number of folds into which data are split.
            past_annotations_limit: TODO: Not sure yet what it exactly does.
            split_sizes: Argument used only if `split_mode == SplitMode.Users`.
                Defines a size of split to divide data into.
            use_finetuned_embeddings: Whether to finetune embeddings.
            test_fold_index: The index of test fold.
            min_annotations_per_user_in_fold: If not none filter out annotators who:
                have less than `min_annotations_per_user_in_fold` annotations in each fold
                or have less than one annotations per each (class_dim, fold) pair
            seed: Will be removed.
            use_cuda: Will be removed.
        """
        super().__init__()

        self.batch_size = batch_size
        self.split_mode = split_mode
        self.embeddings_type = embeddings_type
        self.major_voting = major_voting
        self.test_major_voting = test_major_voting
        self.folds_num = folds_num
        self.past_annotations_limit = past_annotations_limit
        self.use_cuda = use_cuda

        self._test_fold_index = test_fold_index
        self.use_finetuned_embeddings = use_finetuned_embeddings
        self.min_annotations_per_user_in_fold = min_annotations_per_user_in_fold

        self.split_sizes = (
            split_sizes if split_sizes is not None else [0.55, 0.15, 0.15, 0.15]
        )

        # TODO: Shouldn't be called here
        seed_everything(seed)

        self.data, self.annotations = self.load_data_and_annotations()
        self.setup()

    # TODO: Rewrite embeddings
    def _create_embeddings(self) -> None:
        texts = self.data["text"].tolist()
        embeddings_path = self.embeddings_path

        if self.embeddings_type in TRANSFORMER_MODEL_STRINGS:
            model_name = TRANSFORMER_MODEL_STRINGS[self.embeddings_type]
        else:
            model_name = self.embeddings_type

        use_cuda = self.use_cuda and torch.cuda.is_available()

        create_embeddings(
            texts, embeddings_path, model_name=model_name, use_cuda=use_cuda
        )

    def setup(self, stage: Optional[str] = None) -> None:
        annotations = self.annotations
        self._original_annotations = annotations.copy()
        self._split_data()

        if self.past_annotations_limit is not None:
            self.limit_past_annotations(self.past_annotations_limit)

        if self.min_annotations_per_user_in_fold is not None:
            self.filter_annotators()

        self.annotator_biases = self.compute_annotator_biases()

        if self.major_voting:
            self.compute_major_votes()

        if not self.embeddings_path.exists():
            self._create_embeddings()

        embeddings_path = self.embeddings_path

        if self.use_finetuned_embeddings:
            finetune_datamodule_embeddings(self)
            embeddings_path = (
                f"{self.data_dir}/embeddings/{self.embeddings_type}_{self.test_fold}.p"
            )

        with open(embeddings_path, "rb") as f:
            text_idx_to_emb = pickle.load(f)

        embeddings = []
        for text_id in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_id])

        embeddings = np.array(embeddings)

        assert len(self.data.index) == len(embeddings)

        self.text_embeddings = torch.tensor(embeddings)

    def _split_data(self) -> None:
        """Split data into train, validation & test.

        IMPORTANT: This method only assigns columns `"split"` to `self.annotations`.
        Possible values of "split" column:
            train
            val
            test
            none: when text is not annotated

        """
        if self.split_mode == SplitMode.PREDEFINED:
            if "split" not in self.annotations.columns:
                raise Exception(
                    "Split mode {0} is used but no column split in {1}".format(
                        self.split_mode.value,
                        self.annotations.columns,
                    )
                )
            return
        elif self.split_mode == SplitMode.TEXTS:
            if "fold" not in self.annotations.columns:
                raise Exception(
                    "Split mode {0} is used but no column fold in {1}".format(
                        self.split_mode.value,
                        self.annotations.columns,
                    )
                )
            val_fold_index = self.val_fold_index
            test_fold_index = self.test_fold_index
            self.annotations["split"] = "train"
            self.annotations.loc[self.annotations.fold == val_fold_index, "split"] = "val"
            self.annotations.loc[
                self.annotations.fold == test_fold_index, "split"
            ] = "test"
        elif self.split_mode == SplitMode.USERS:
            # TODO: Whats going on here?
            self.data = split_texts(self.data, self.split_sizes)

            text_id_to_text_split = self.data.set_index("text_id")["text_split"]
            text_id_to_text_split = text_id_to_text_split.to_dict()

            annotator_id_to_fold = self.annotations.set_index("annotator_id")["fold"]
            annotator_id_to_fold = annotator_id_to_fold.to_dict()

            def _get_annotation_split(row):
                text_id, annotator_id = row["text_id"], row["annotator_id"]
                text_split = text_id_to_text_split[text_id]
                annotator_fold = annotator_id_to_fold[annotator_id]

                if text_split == "past":
                    return "train"
                if text_split == "present" and annotator_fold in self.train_folds_indices:
                    return "train"
                if text_split == "future1" and annotator_fold == self.val_fold_index:
                    return "val"
                if text_split == "future2" and annotator_fold == self.test_fold_index:
                    return "test"

                return "none"

            self.annotations["split"] = "none"
            self.annotations["split"] = self.annotations[
                ["text_id", "annotator_id"]
            ].apply(_get_annotation_split, axis=1)
        else:
            raise Exception(
                "Split mode {0} is invalid".format(
                    self.split_mode.value,
                )
            )

    def compute_major_votes(self) -> None:
        """Computes mean votes for texts in train folds."""
        annotations = self.annotations
        annotation_columns = self.annotation_columns

        text_id_to_fold = (
            annotations.loc[:, ["text_id", "fold"]]
            .drop_duplicates()
            .set_index("text_id")
            .to_dict()["fold"]
        )

        text_id_to_split = (
            annotations.loc[:, ["text_id", "split"]]
            .drop_duplicates()
            .set_index("text_id")
            .to_dict()["split"]
        )

        val_test_annotations = None
        if not self.test_major_voting:
            val_test_annotations = annotations.loc[
                annotations.split.isin(["val", "test"])
            ]
            annotations = annotations.loc[~annotations.split.isin(["val", "test"])]

        dfs = []
        for col in annotation_columns:
            aggregate_lambda = lambda x: pd.Series.mode(x)[0]

            dfs.append(annotations.groupby("text_id")[col].apply(aggregate_lambda))

        annotations = pd.concat(dfs, axis=1).reset_index()
        annotations["annotator_id"] = 0
        annotations["split"] = annotations["text_id"].map(text_id_to_split)
        annotations["fold"] = annotations["text_id"].map(text_id_to_fold)

        if not self.test_major_voting:
            self.annotations = pd.concat([annotations, val_test_annotations])
        else:
            self.annotations = annotations

    def compute_annotator_biases(self) -> pd.DataFrame:
        annotations_with_data = self.annotations_with_data

        if self.split_mode == SplitMode.USERS:
            personal_df_mask = annotations_with_data.text_split == "past"
        else:
            personal_df_mask = annotations_with_data.split == "train"

        personal_df = annotations_with_data.loc[personal_df_mask]

        annotation_columns = self.annotation_columns
        annotator_biases = get_annotator_biases(personal_df, annotation_columns)

        all_annotator_ids = self._original_annotations.annotator_id.unique()
        annotator_id_df = pd.DataFrame(all_annotator_ids, columns=["annotator_id"])

        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how="left"
        )
        return annotator_biases.set_index("annotator_id").sort_index().fillna(0)

    def train_dataloader(self) -> DataLoader:
        """Returns dataloader for training part of the dataset.

        Returns:
            DataLoader: training dataloader for the dataset.
        """
        return self._get_dataloader("train", True)

    def val_dataloader(self) -> DataLoader:
        """Returns dataloader for validation part of the dataset.

        Returns:
            DataLoader: validation dataloader for the dataset.
        """
        return self._get_dataloader("val", False)

    def test_dataloader(self) -> DataLoader:
        """Returns dataloader for testing part of the dataset.

        Returns:
            DataLoader: testing dataloader for the dataset.
        """
        return self._get_dataloader("test", False)

    def custom_dataloader(
        self, split_name: str = "none", shuffle: bool = False
    ) -> DataLoader:
        return self._get_dataloader(split_name, shuffle)

    def _get_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        annotations = self.annotations
        annotations = annotations.loc[annotations.split == split]

        X, y = self._get_data_by_split(annotations)
        # TODO: X shouldn't be an array to avoid magic numbers
        text_ids = X[:, 0]
        annotator_ids = X[:, 1]
        dataset = TextFeaturesBatchDataset(
            text_ids=text_ids,
            annotator_ids=annotator_ids,
            embeddings=self.text_embeddings,
            raw_texts=self.data["text"].values,
            annotator_biases=self.annotator_biases.values.astype(float),
            y=y,
        )

        if shuffle:
            order_sampler_cls = torch.utils.data.sampler.RandomSampler
        else:
            order_sampler_cls = torch.utils.data.sampler.SequentialSampler

        batch_size = self.batch_size
        num_annotations = len(annotations.index)
        batch_size = min(batch_size, int(num_annotations / 15))
        batch_size = max(batch_size, 1)

        sampler = torch.utils.data.sampler.BatchSampler(
            order_sampler_cls(dataset),
            batch_size=batch_size,
            drop_last=False,
        )

        return torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=None, num_workers=4
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

    def _get_data_by_split(
        self, annotations: pd.DataFrame
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
        df = annotations

        X = df.loc[:, ["text_id", "annotator_id"]]
        y = df[self.annotation_columns]

        X, y = X.values, y.values

        if y.ndim < 2:
            y = y[:, None]

        return X, y

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

    def filter_annotators(self) -> None:
        """Filters annotators with less than `min_annotations_per_user_in_fold` annotations
        in each fold and with less than one annotations per each (class_dim, fold) pair.
        """
        if self.split_mode == SplitMode.USERS:
            raise Exception("Cannot use user folds with min_annotations_per_user_in_fold")

        min_annotations = self.min_annotations_per_user_in_fold

        annotation_counts = self.annotations.loc[
            :, ["annotator_id", "fold"]
        ].value_counts()
        annotation_counts = annotation_counts.reset_index().rename(
            columns={0: "annotation_number"}
        )
        annotators_to_ignore = annotation_counts.loc[
            annotation_counts.annotation_number < min_annotations
        ].annotator_id.unique()

        self.annotations = self.annotations[
            ~self.annotations.annotator_id.isin(annotators_to_ignore)
        ]

        annotations = self.annotations
        annotation_columns = self.annotation_columns
        folds_num = self.folds_num
        min_class_annotations = 1

        annotators_to_filter = set()

        for annotation_column in annotation_columns:
            class_dim = annotations[annotation_column].nunique()

            annotations_per_class = annotations.loc[
                :, ["annotator_id", "fold", annotation_column]
            ].value_counts()

            annotations_per_class = annotations_per_class[
                annotations_per_class >= min_class_annotations
            ]

            class_fold_per_annotator = (
                annotations_per_class.reset_index().annotator_id.value_counts()
            )

            annotators_to_filter.update(
                class_fold_per_annotator[
                    class_fold_per_annotator < folds_num * class_dim
                ].index.tolist()
            )

        self.annotations = self.annotations.loc[
            ~self.annotations.annotator_id.isin(annotators_to_filter)
        ]
