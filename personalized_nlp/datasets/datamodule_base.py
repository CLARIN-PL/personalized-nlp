import abc
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.controversy import get_conformity
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.utils.embeddings import create_embeddings
from personalized_nlp.utils.finetune import finetune_datamodule_embeddings
from pytorch_lightning import LightningDataModule, seed_everything
from settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# TODO specify types!
# TODO add docstring!
class BaseDataModule(LightningDataModule, abc.ABC):
    @abc.abstractproperty
    def class_dims(self) -> List[int]:
        raise NotImplementedError()

    @abc.abstractproperty
    def annotation_columns(self) -> List[str]:
        raise NotImplementedError()

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"embeddings/text_id_to_emb_{self.embeddings_type}.p"

    @property
    def annotations_file(self) -> str:
        return f"annotations_{self.stratify_folds_by}_folds.csv"

    @property
    def data_file(self) -> str:
        return f"data_processed.csv"

    @abc.abstractproperty
    def data_dir(self) -> Path:
        raise NotImplementedError()

    @property
    def annotators_number(self) -> int:
        return max(self.annotations["annotator_id"]) + 1

    @property
    def train_text_split_names(self) -> List[str]:
        return ["present", "past"]

    @property
    def val_text_split_names(self) -> List[str]:
        return ["future1"]

    @property
    def test_text_split_names(self) -> List[str]:
        return ["future2"]

    @property
    def train_folds(self) -> List[int]:
        all_folds = range(self.folds_num)
        exclude_folds = [self.val_fold, self.test_fold]

        return [fold for fold in all_folds if fold not in exclude_folds]

    @property
    def num_annotations(self) -> int:
        return len(self.annotations)

    @property
    def val_fold(self) -> int:
        return (self._test_fold + 1) % self.folds_num

    @property
    def test_fold(self) -> int:
        return self._test_fold

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
        embeddings_type: str = "labse",
        major_voting: bool = False,
        test_major_voting: bool = False,
        folds_num: int = 10,
        regression: bool = False,
        past_annotations_limit: Optional[int] = None,
        stratify_folds_by: Optional[str] = "users",
        split_sizes: Optional[List[str]] = None,
        use_finetuned_embeddings: bool = False,
        test_fold: Optional[int] = None,
        min_annotations_per_user_in_fold: Optional[int] = None,
        seed: int = 22,
        use_cuda: bool = False,
        **kwargs,
    ):
        """_summary_
        Args:
            batch_size (int, optional): Batch size for data loaders. Defaults to 3000.
            embeddings_type (str, optional): string identifier of embedding. Defaults to "labse".
            major_voting (bool, optional): if true, use major voting. Defaults to False.
            test_major_voting (bool, optional): if true, use major voting also on val and test dataset parts. Defaults to False.
            folds_num (int, optional): Number of folds. Defaults to 10.
            regression (bool, optional): Normalize labels to [0, 1] range with min-max method. Defaults to False.
            past_annotations_limit (Optional[int], optional): Maximum number of annotations in past dataset part. Defaults to None.
            stratify_folds_by (str, optional): How to stratify annotations: 'texts' or 'users'. Defaults to 'texts'.
            use_finetuned_embeddings (bool, optional): if true, use finetuned embeddings. Defaults to False.
            filtered_annotations (str, optional): path to dataframe for filter annotations (for example choose only user=1, text=1,2,4 etc). Defaults to None.
        """

        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=dims,
        )

        self._init_args = locals()
        del self._init_args["self"]
        del self._init_args["__class__"]

        self.batch_size = batch_size
        self.embeddings_type = embeddings_type
        self.major_voting = major_voting
        self.test_major_voting = test_major_voting
        self.folds_num = folds_num
        self.regression = regression
        self.past_annotations_limit = past_annotations_limit
        self.stratify_folds_by = stratify_folds_by
        self.use_cuda = use_cuda

        self._test_fold = test_fold if test_fold is not None else 0
        self.use_finetuned_embeddings = use_finetuned_embeddings
        self.min_annotations_per_user_in_fold = min_annotations_per_user_in_fold

        self.split_sizes = (
            split_sizes if split_sizes is not None else [0.55, 0.15, 0.15, 0.15]
        )

        self.annotations = pd.DataFrame([])
        self.data = pd.DataFrame([])

        seed_everything(seed)

        os.makedirs(self.data_dir / "embeddings", exist_ok=True)

        self.prepare_data()
        self.setup()

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

        # self._assign_folds()
        self._assign_splits()

        if self.past_annotations_limit is not None:
            self.limit_past_annotations(self.past_annotations_limit)

        if self.min_annotations_per_user_in_fold is not None:
            self.filter_annotators()

        if not os.path.exists(self.embeddings_path):
            self._create_embeddings()

        embeddings_path = self.embeddings_path

        if self.use_finetuned_embeddings:
            finetune_datamodule_embeddings(self)
            embeddings_path = f"{self.data_dir}/embeddings/{self.embeddings_type}_{self.test_fold}_{self.stratify_folds_by}.p"

        with open(embeddings_path, "rb") as f:
            text_idx_to_emb = pickle.load(f)

        embeddings = []
        for text_id in range(len(text_idx_to_emb.keys())):
            embeddings.append(text_idx_to_emb[text_id])

        embeddings = np.array(embeddings)

        assert len(self.data.index) == len(embeddings)

        self.text_embeddings = torch.tensor(embeddings)

        self._compute_annotator_stats()

        if self.regression:
            self._normalize_labels()

        if self.major_voting:
            self.compute_major_votes()

        self._after_setup()

    def _after_setup(self):
        pass

    def _assign_splits(self) -> None:
        if "split" in self.annotations.columns:
            return

        if self.stratify_folds_by == "texts":
            val_fold = self.val_fold
            test_fold = self.test_fold

            self.annotations["split"] = "train"
            self.annotations.loc[self.annotations.fold == val_fold, "split"] = "val"
            self.annotations.loc[self.annotations.fold == test_fold, "split"] = "test"
        else:
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
                if text_split == "present" and annotator_fold in self.train_folds:
                    return "train"
                if text_split == "future1" and annotator_fold == self.val_fold:
                    return "val"
                if text_split == "future2" and annotator_fold == self.test_fold:
                    return "test"

                return "none"

            self.annotations["split"] = "none"
            self.annotations["split"] = self.annotations[
                ["text_id", "annotator_id"]
            ].apply(_get_annotation_split, axis=1)

    def _assign_folds(self):
        """Randomly assign fold to each annotation."""
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

    def _normalize_labels(self):
        annotation_columns = self.annotation_columns
        df = self.annotations

        mins = df.loc[:, annotation_columns].values.min(axis=0)
        df.loc[:, annotation_columns] = df.loc[:, annotation_columns] - mins

        maxes = df.loc[:, annotation_columns].values.max(axis=0)
        df.loc[:, annotation_columns] = df.loc[:, annotation_columns] / maxes

        df.loc[:, annotation_columns] = df.loc[:, annotation_columns].fillna(0)

    def compute_major_votes(self) -> None:
        """Computes mean votes for texts in train folds."""
        annotations = self.annotations
        annotation_columns = self.annotation_columns
        id_col = "text_id" if self.stratify_folds_by == "text" else "annotator_id"

        id_to_fold = (
            annotations.loc[:, [id_col, "fold"]]
            .drop_duplicates()
            .set_index(id_col)
            .to_dict()["fold"]
        )

        annotations_to_concat = []
        for split in annotations.split.unique():
            split_annotations = annotations.loc[annotations.split == split]

            if split == "none":
                annotations_to_concat.append(split_annotations)
                continue

            if split in ["val", "test"] and not self.test_major_voting:
                annotations_to_concat.append(split_annotations)
                continue

            dfs = []
            for col in annotation_columns:
                if self.regression:
                    aggregate_lambda = lambda x: x.mean()
                else:
                    aggregate_lambda = lambda x: pd.Series.mode(x).tolist()[-1]

                dfs.append(
                    split_annotations.groupby("text_id")[col].apply(aggregate_lambda)
                )

            major_voted_annotations = pd.concat(dfs, axis=1).reset_index()
            major_voted_annotations["annotator_id"] = 0
            major_voted_annotations["split"] = split
            major_voted_annotations["fold"] = annotations[id_col].map(id_to_fold)

            annotations_to_concat.append(major_voted_annotations)

        self.annotations = pd.concat(annotations_to_concat)

    def compute_annotator_biases(self):
        annotations_with_data = self.annotations.merge(
            self.data[["text_id", "text_split"]]
        )

        if self.stratify_folds_by == "users":
            personal_df_mask = annotations_with_data.text_split == "past"
        else:
            personal_df_mask = annotations_with_data.split == "train"

        personal_df = annotations_with_data.loc[personal_df_mask]
        all_annotator_ids = self._original_annotations.annotator_id.unique()
        annotator_id_df = pd.DataFrame(all_annotator_ids, columns=["annotator_id"])

        annotator_biases = get_annotator_biases(personal_df, annotation_columns)
        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how="left"
        )
        self.annotator_biases = (
            annotator_biases.set_index("annotator_id").sort_index().fillna(0)
        )

        if not self.regression:
            annotator_conformities = get_conformity(personal_df, annotation_columns)
            annotator_conformities = annotator_id_df.merge(
                annotator_conformities.reset_index(), how="left"
            )
            self.annotator_conformities = (
                annotator_conformities.set_index("annotator_id").sort_index().fillna(0)
            )
        else:
            self.annotator_conformities = self.annotator_biases

        self.annotator_mean_text_embeddings = self.get_annotator_mean_text_embeddings(
            personal_df
        )

    def get_annotator_mean_text_embeddings(
        self, personal_df: pd.DataFrame
    ) -> np.ndarray:
        annotation_col = self.annotation_columns[0]
        all_annotator_ids = self.annotations[["annotator_id"]].drop_duplicates()

        positive_text_df = personal_df.loc[personal_df[annotation_col] == 1]
        negative_text_df = personal_df.loc[personal_df[annotation_col] == 0]

        def _get_embeddings(df):
            mean_embeddings_df = df.groupby(["annotator_id"])["text_id"].apply(
                lambda text_ids: self.text_embeddings[text_ids.values]
                .numpy()
                .mean(axis=0)
            )

            mean_embeddings_df = mean_embeddings_df.reset_index()
            mean_embeddings_df = mean_embeddings_df.merge(
                all_annotator_ids, how="right"
            )
            embeddings_dict = mean_embeddings_df.set_index("annotator_id").to_dict()
            embeddings_dict = embeddings_dict["text_id"]
            embeddings_dict = {
                k: v if isinstance(v, np.ndarray) else np.zeros(self.text_embedding_dim)
                for k, v in embeddings_dict.items()
            }
            embedding_list = [embeddings_dict[i] for i in range(len(all_annotator_ids))]
            return np.vstack(embedding_list)

        positive_embeddings = _get_embeddings(positive_text_df)
        negative_embeddings = _get_embeddings(negative_text_df)

        return np.hstack([positive_embeddings, negative_embeddings])

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
        text_features = self._get_text_features()
        annotator_features = self._get_annotator_features()

        dataset = BatchIndexedDataset(
            X,
            y,
            text_features=text_features,
            annotator_features=annotator_features,
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

    def _get_annotator_features(self) -> Dict[str, np.ndarray]:
        """Returns dictionary of features of all annotators in the dataset.
        Each feature should be a numpy array of whatever dtype, with length
        equal to number of annotators in the dataset. Features can be used by
        models during training.
        :return: dictionary of annotator features
        :rtype: Dict[str, Any]
        """
        data = {
            "annotator_biases": self.annotator_biases.values.astype(float),
            "conformity": self.annotator_conformities.values.astype(float),
            "annotator_mean_text_embeddings": self.annotator_mean_text_embeddings,
        }

        return data

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

        # X["text_id"] = X["text_id"].apply(lambda r_id: self.text_id_idx_dict[r_id])
        # X["annotator_id"] = X["annotator_id"].apply(
        #     lambda w_id: self.annotator_id_idx_dict[w_id]
        # )

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
        annotations = annotations
        embeddings = self.text_embeddings.to("cpu").numpy()

        # annotations["text_idx"] = annotations["text_id"].apply(
        #     lambda r_id: self.text_id_idx_dict[r_id]
        # )
        # annotations["annotator_idx"] = annotations["annotator_id"].apply(
        #     lambda w_id: self.annotator_id_idx_dict[w_id]
        # )

        X = np.vstack([embeddings[i] for i in annotations["text_id"].values])
        y = annotations[self.annotation_columns].values

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
        in each fold and with less than one annotation per each (class_dim, fold) pair.
        """
        if self.stratify_folds_by == "users":
            raise Exception(
                "Cannot use user folds with min_annotations_per_user_in_fold"
            )

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
