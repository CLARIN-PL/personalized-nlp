import abc
from enum import Enum
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, seed_everything

from torch.utils.data import DataLoader

from personalized_active_learning.datamodules.types import TextFeaturesBatchDataset
from personalized_active_learning.embeddings import EmbeddingsCreator
from personalized_active_learning.embeddings.finetune import fine_tune_embeddings
from personalized_active_learning.embeddings.personalised.base import (
    PersonalisedEmbeddings
)
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.data_splitting import split_texts


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


class BaseDataModule(LightningDataModule, abc.ABC):
    """The base class from each dataset class should derive."""

    def __init__(
        self,
        embeddings_creator: EmbeddingsCreator,
        personalized_embeddings_cls: Optional[PersonalisedEmbeddings] = None,
        batch_size: int = 3000,
        split_mode: SplitMode = SplitMode.TEXTS,
        subset_ratio: float = 1.0,
        major_voting: bool = False,  # TODO: Not sure if we need that
        test_major_voting: bool = False,  # TODO: Not sure if we need that
        folds_num: int = 10,
        past_annotations_limit: Optional[int] = None,
        split_sizes: Optional[
            List[str]
        ] = None,  # TODO: Used only when split mode is user, ugly
        use_finetuned_embeddings: bool = False,
        test_fold_index: int = 0,
        min_annotations_per_user_in_fold: Optional[int] = None,
        seed: int = 22,  # TODO: It shouldn't be use by datamodule but higher!!!
    ):
        """Initialize object.

        Args:
            embeddings_creator: The Embeddings creator.
            use_finetuned_embeddings: If `True` the embeddings will be finetuned.
            batch_size: The size of batch.
            split_mode: The split mode used to divide data into training, val & test.
            subset_ratio: Generate subset from the dataset with specified ratio.
                Ratio should be in (0.0, 1.0>, where 1.0 means whole dataset.
            major_voting: If `True` do not use personalization on training data.
                I.E. Text annotated by 5 annotators will be transformed to single text.
            test_major_voting: If `True` do not use personalization on val & test data.
                I.E. Text annotated by 5 annotators will be transformed to single text.
            folds_num: The number of folds into which data are split.
            past_annotations_limit: This variable limits number of annotatinos per user
                in `past` dataset. Used in SplitMode.Users.
            split_sizes: Argument used only if `split_mode == SplitMode.Users`.
                Defines a size of split to divide data into.
            test_fold_index: The index of test fold.
            min_annotations_per_user_in_fold: If not none filter out annotators who:
                have less than `min_annotations_per_user_in_fold` annotations in each fold
                or have less than one annotations per each (class_dim, fold) pair
            personalized_embeddings_type: Value indicates usage of personalized data
                in embeddings. Currently available options are:
                    - "": no personalisation.
                    - "annotator_id": add `annotator_id` to texts.
            seed: Will be removed.

        """
        super().__init__()
        # TODO: Ugly, needed for fine-tuning embeddings
        self.init_kwargs = locals()
        del self.init_kwargs["self"]
        del self.init_kwargs["__class__"]

        self.embeddings_creator = embeddings_creator
        # Need to reset the name, because embeddings_creator is defined in a dict once
        self.embeddings_creator.set_personalised_embeddings_name("")
        self.batch_size = batch_size
        self.split_mode = split_mode
        self.major_voting = major_voting
        self.test_major_voting = test_major_voting
        self.folds_num = folds_num
        self.past_annotations_limit = past_annotations_limit

        self._test_fold_index = test_fold_index
        self.subset_ratio = subset_ratio
        self.use_finetuned_embeddings = use_finetuned_embeddings
        self.min_annotations_per_user_in_fold = min_annotations_per_user_in_fold
        self.personalized_embeddings_cls = personalized_embeddings_cls

        self._raw_text: np.ndarray  # Moved to self.setup

        # TODO: Move default value to same constants
        self.split_sizes = (
            split_sizes if split_sizes is not None else [0.55, 0.15, 0.15, 0.15]
        )
        seed_everything(seed)

        self.data, self.annotations = self.load_data_and_annotations()
        self.setup()

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

    @property
    @abc.abstractmethod
    def annotation_columns(self) -> List[str]:
        """Get the annotation columns.

        Each collum corresponds to separated supervised task, e.g.:
        [`humor`, `sarcasm`]

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
        """Get the index of test fold.

        Returns:
            The index of test fold.

        """
        return self._test_fold_index

    @property
    def annotations_with_data(self) -> pd.DataFrame:
        """Get the annotations and data merged to single `DataFrame`.

        Returns:
            The `DataFrame` containing both data & annotations.

        """
        return self.annotations.merge(self.data)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup DataSet.

        Args:
            stage: Left for compatibility with pytorch lighting.

        Prepare data to use. Currently this method:
            1. Checks if split mode has valid parameters
            2. Apply personalisation to the text dataset
            3. Finetunes text embeddings if needed.
            4. Creates subset of data if needed
            5. Splits data into training, validation, test sets.
            6. Limits annotations if needed.
            7. Filters annotations if needed.
            8. Computes annotators biases.
            9. Applies major voting mechanism if needed.
        """

        self._validate_split_mode()
        self.data, self.annotations = self._apply_personalised_embeddings()
        self._raw_text = self.data["text"].values

        if self.use_finetuned_embeddings:  # TODO: move to embedings?
            # TODO: Ugly hack but we probably don't have time to change that
            fine_tune_embeddings(self)

        texts = self.data["text"].tolist()
        self.text_embeddings = self.embeddings_creator.get_embeddings(texts=texts)
        assert len(self.data.index) == len(self.text_embeddings)

        self.data, self.annotations = self._generate_subset()
        self._original_annotations = self.annotations.copy()
        self._split_data()

        if self.past_annotations_limit is not None:
            self._limit_past_annotations(self.past_annotations_limit)
        if self.min_annotations_per_user_in_fold is not None:
            self._filter_annotators()

        self.annotator_biases = self._compute_annotator_biases()

        if self.major_voting:
            self.compute_major_votes()

    def _apply_personalised_embeddings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply personalisation to text data with `self.personalized_embeddings_cls`

        Modify `self.data.text` with user informations with rules defined in
        `PersonalisedEmbedings` based classes. Modify `self.annotations` if neccessary

        Returns:
            pd.DataFrame: Personalised texts in `self.data` and modified
                `self.annotations`
        """

        if not self.personalized_embeddings_cls:
            return self.data, self.annotations

        personalised_embeddings = self.personalized_embeddings_cls(
            self.data,
            self.annotations
        )

        logging.info(
            "Preparing personalised embeddings: %s" % personalised_embeddings.name
        )

        self.embeddings_creator.set_personalised_embeddings_name(
            personalised_embeddings.name
        )

        return personalised_embeddings.apply_personalisation()

    def _generate_subset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate subset of dataset

        Sample `self.data` and `self.annotations` with `self.subset_ratio`.

        Raises:
            Exception: Invalid split mode
            Exception: Missing column for a specific split mode

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: New data and annotations
        """

        if self.subset_ratio <= 0 or self.subset_ratio >= 1:
            return self.data, self.annotations

        if self.split_mode == SplitMode.PREDEFINED:
            sampling_column = "split"
        elif self.split_mode == SplitMode.TEXTS or self.split_mode == SplitMode.USERS:
            sampling_column = "fold"
        else:
            raise Exception(
                "Split mode {0} has not configured sampling_column".format(
                    self.split_mode.value,
                )
            )

        sampling_column_values = self.annotations[sampling_column].unique()
        sampled_annotations = [
            self.annotations[
                self.annotations[sampling_column] == sampling_value
            ].sample(frac=self.subset_ratio)
            for sampling_value in sampling_column_values
        ]
        annotations = pd.concat(sampled_annotations, ignore_index=True)
        data = self.data[
            self.data.text_id.isin(annotations.text_id.unique())
        ].reset_index(drop=True)

        return data, annotations

    def _validate_split_mode(self) -> None:
        """Validate if split parameters are valid

        Check if `self.split_mode` has necessary columns in `self.annotations`
        `SplitMode.PREDEFINED`: `split` column
        `SplitMode.TEXTS`, `SplitMode.USERS`: `fold` column

        Raises:
            Exception: No split column in `self.annotations`
            Exception: No fold column in `self.annotations`
            Exception: Invalid split mode
        """

        if self.split_mode == SplitMode.PREDEFINED:
            if "split" not in self.annotations.columns:
                raise Exception(
                    "Split mode {0} is used but no split column in {1}".format(
                        self.split_mode.value,
                        self.annotations.columns,
                    )
                )
        elif self.split_mode == SplitMode.TEXTS or self.split_mode == SplitMode.USERS:
            if "fold" not in self.annotations.columns:
                raise Exception(
                    "Split mode {0} is used but no fold column in {1}".format(
                        self.split_mode.value,
                        self.annotations.columns,
                    )
                )
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
            dfs.append(
                annotations.groupby("text_id")[col].apply(lambda x: pd.Series.mode(x)[0])
            )

        annotations = pd.concat(dfs, axis=1).reset_index()
        annotations["annotator_id"] = 0
        annotations["split"] = annotations["text_id"].map(text_id_to_split)
        annotations["fold"] = annotations["text_id"].map(text_id_to_fold)

        if not self.test_major_voting:
            self.annotations = pd.concat([annotations, val_test_annotations])
        else:
            self.annotations = annotations

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
        self,
        split_name: str = "none",
        shuffle: bool = False,
    ) -> DataLoader:
        """Get a custom dataloader.

        Used purely to obtain a dataloader with split `none`.

        Args:
            split_name: The name of selected split. Must be one of:
                train, val, test, none.
            shuffle: Whether to shuffle data.

        Returns:
            A dataloader with specified parameters.

        """
        return self._get_dataloader(split_name, shuffle)

    def _compute_annotator_biases(self) -> pd.DataFrame:
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

    def _get_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        """Get a dataloader.

        Args:
            split_name: The name of selected split. Must be one of:
                train, val, test, none.
            shuffle: Whether to shuffle data.

        Returns:
            A dataloader with specified parameters.

        """
        annotations = self.annotations
        annotations = annotations.loc[annotations.split == split]

        data, y = self.get_data_and_labels(annotations)
        # TODO: X shouldn't be an array to avoid magic numbers
        text_ids = data["text_id"]
        annotator_ids = data["annotator_id"]
        dataset = TextFeaturesBatchDataset(
            text_ids=text_ids.values,
            annotator_ids=annotator_ids.values,
            embeddings=self.text_embeddings,
            raw_texts=self._raw_text,
            annotator_biases=self.annotator_biases.values.astype(float),
            y=y,
        )

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = self.batch_size
        num_annotations = len(annotations.index)
        batch_size = min(batch_size, int(num_annotations / 15))
        batch_size = max(batch_size, 1)

        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
        )

        return torch.utils.data.DataLoader(
            dataset,
            sampler=batch_sampler,
            batch_size=None,
            num_workers=4,  # TODO: Number of workers shouldn't be hardcoded
        )

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
            return
        elif self.split_mode == SplitMode.TEXTS:
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

    def get_data_and_labels(
        self, annotations: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """

        Args:
            annotations: The annotations of texts.

        Returns: Pair of:
            Data: I.E. DataFrame containing text ids and annotator ids.
            Labels: I.E. Numpy array containing labels.

        """
        df = annotations

        X = df.loc[:, ["text_id", "annotator_id"]]
        y = df[self.annotation_columns]

        y = y.values

        if y.ndim < 2:
            y = y[:, None]

        return X, y

    def _limit_past_annotations(self, limit: int):
        # TODO: What this method does?
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

    def _filter_annotators(self) -> None:
        """Filter out annotators with insignificant number of annotations.

        TODO: Change it to return something instead of modifying class variable
        IMPORTANT: This method modifies `self.annotations`.

        Filter out annotators who:
            have less than `min_annotations_per_user_in_fold` annotations in fold.
            have zero annotations in one of pairs (class_dim, fold).

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
        min_class_annotations = 1

        annotators_to_filter = set()

        for annotation_column in self.annotation_columns:
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
                    class_fold_per_annotator < self.folds_num * class_dim
                ].index.tolist()
            )

        self.annotations = self.annotations.loc[
            ~self.annotations.annotator_id.isin(annotators_to_filter)
        ]
