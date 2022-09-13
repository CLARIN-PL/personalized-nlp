from typing import Tuple, List, Optional, Dict, Any
import os
import pickle
from pathlib import Path


import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, seed_everything

from settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS, DATA_DIR
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.controversy import get_conformity
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.utils.embeddings import create_embeddings
from personalized_nlp.utils.finetune import finetune_datamodule_embeddings


class CartographyDataModule(LightningDataModule):
    
    def __init__(
        self,
        # new
        annotations: pd.DataFrame, 
        data: pd.DataFrame, 
        test_fold: int = 0, 
        val_fold: int = 1,
        
        # old
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dims=None,
        batch_size: int = 3000,
        embeddings_type: str = "labse",
        major_voting: bool = False,
        regression: bool = False,
        past_annotations_limit: Optional[int] = None,
        use_finetuned_embeddings: bool = False,
        min_annotations_per_user_in_fold: Optional[int] = None,
        seed: int = 22,
        **kwargs,
    ):
        """_summary_
        Args:
            batch_size (int, optional): Batch size for data loaders. Defaults to 3000.
            embeddings_type (str, optional): string identifier of embedding. Defaults to "bert".
            major_voting (bool, optional): if true, use major voting. Defaults to False.
            folds_num (int, optional): Number of folds. Defaults to 10.
            regression (bool, optional): Normalize labels to [0, 1] range with min-max method. Defaults to False.
            past_annotations_limit (Optional[int], optional): Maximum number of annotations in past dataset part. Defaults to None.
            stratify_folds_by (str, optional): How to stratify annotations: 'texts' or 'users'. Defaults to 'texts'.
            use_finetuned_embeddings (bool, optional): if true, use finetuned embeddings. Defaults to False.
            filtered_annotations (str, optional): path to dataframe for filter annotations (for example choose only user=1, text=1,2,4 etc). Defaults to None.
        """

        super(CartographyDataModule, self).__init__(
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
        self.regression = regression
        self.past_annotations_limit = past_annotations_limit

       # self._test_fold = test_fold if test_fold is not None else 0
        self.use_finetuned_embeddings = use_finetuned_embeddings
        self.min_annotations_per_user_in_fold = min_annotations_per_user_in_fold


        self.annotations = annotations
        self.data = data
        self._test_fold = test_fold
        self._val_fold = val_fold

        seed_everything(seed)

        self.prepare_data()
        self.setup()
        
        # toxicity
        os.makedirs(self.data_dir / "embeddings", exist_ok=True)
    
    def add_top_k(self, predictions: pd.DataFrame, metric: str, amount: int, ascending: bool = True) -> None:
        predictions.sort_values(by=metric, inplace=True, ascending=ascending)
        to_add = predictions.head(n=amount)
        self.annotations.loc[self.annotations['guid'].isin(to_add['guid']), 'available'] = True
    
    def prepare_data(self) -> None:
        self.data = self._remap_column_names(self.data)
        self.data["text"] = self.data["text"].str.replace("NEWLINE_TOKEN", "  ")
        self.annotations = self._remap_column_names(self.annotations)
        
        self.annotations["split"] = "train"
        self.annotations.loc[self.annotations.fold == self.val_fold, "split"] = "val"
        self.annotations.loc[self.annotations.fold == self.test_fold, "split"] = "test"


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
        df = annotations[annotations['available'] == True]

        X = df.loc[:, ["text_id", "annotator_id"]]
        y = df[self.annotation_columns]


        X, y = X.values, y.values

        if y.ndim < 2:
            y = y[:, None]

        return X, y


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

    def setup(self, stage: Optional[str] = None) -> None:
        annotations = self.annotations
        self._original_annotations = annotations.copy()


        self.compute_annotator_biases()

        if self.regression:
            self._normalize_labels()

        if self.major_voting:
            self.compute_major_votes()

        if not os.path.exists(self.embeddings_path):
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


    def _normalize_labels(self):
        annotation_columns = self.annotation_columns
        df = self.annotations

        mins = df.loc[:, annotation_columns].values.min(axis=0)
        df.loc[:, annotation_columns] = df.loc[:, annotation_columns] - mins

        maxes = df.loc[:, annotation_columns].values.max(axis=0)
        df.loc[:, annotation_columns] = df.loc[:, annotation_columns] / maxes

        df.loc[:, annotation_columns] = df.loc[:, annotation_columns].fillna(0)


    def compute_annotator_biases(self):
        annotations_with_data = self.annotations_with_data

        personal_df_mask = ~annotations_with_data.fold.isin([self.test_fold, self.val_fold])

        personal_df = annotations_with_data.loc[personal_df_mask]

        annotation_columns = self.annotation_columns
        annotator_biases = get_annotator_biases(personal_df, annotation_columns)

        all_annotator_ids = self._original_annotations.annotator_id.unique()
        annotator_id_df = pd.DataFrame(all_annotator_ids, columns=["annotator_id"])

        annotator_biases = annotator_id_df.merge(
            annotator_biases.reset_index(), how="left"
        )
        self.annotator_biases = (
            annotator_biases.set_index("annotator_id").sort_index().fillna(0)
        )

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

        sampler = torch.utils.data.sampler.BatchSampler(
            order_sampler_cls(dataset),
            batch_size=self.batch_size,
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

        # annotations["text_idx"] = annotations["text_id"].apply(
        #     lambda r_id: self.text_id_idx_dict[r_id]
        # )
        # annotations["annotator_idx"] = annotations["annotator_id"].apply(
        #     lambda w_id: self.annotator_id_idx_dict[w_id]
        # )

        X = np.vstack([embeddings[i] for i in annotations["text_idx"].values])
        y = annotations[self.annotation_columns].values

        return X, y

    def get_conformity(self, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        return get_conformity(self.annotations, self.annotation_columns)


    # toxicity
    def _remap_column_names(self, df):
        mapping = {"rev_id": "text_id", "worker_id": "annotator_id", "comment": "text"}
        df.columns = [mapping.get(col, col) for col in df.columns]
        return df
    
    @property
    def class_dims(self) -> List[int]:
        return [2]

    @property
    def annotation_columns(self) -> List[str]:
        return ["toxicity"]

    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir / f"embeddings/rev_id_to_emb_{self.embeddings_type}_toxic.p"
        )

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / "wiki_data"
    
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
        return self._val_fold

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
    
    @property
    def train_pct_used(self) -> float:
        train_df = self.annotations.loc[self.annotations.split == 'train']
        if (~train_df['available']).sum() == 0:
            return 1.0
        return (train_df['available'] == True).sum() / len(train_df)
    
    @property
    def unused_train(self) -> pd.DataFrame:
        train_df = self.annotations.loc[self.annotations.split == 'train']
        return train_df[train_df['available'] == False].copy()

    @property
    def train_size(self) -> int:
        train_df = self.annotations.loc[self.annotations.split == 'train']
        return len(train_df)

    # do wywalenia
     # def compute_major_votes(self) -> None:
    #     """Computes mean votes for texts in train folds."""
    #     annotations = self.annotations
    #     annotation_columns = self.annotation_columns

    #     text_id_to_fold = (
    #         annotations.loc[:, ["text_id", "fold"]]
    #         .drop_duplicates()
    #         .set_index("text_id")
    #         .to_dict()["fold"]
    #     )

    #     val_test_annotations = annotations.loc[annotations.fold.isin([self.test_fold, self.val_fold])]
    #     annotations = annotations.loc[~annotations.fold.isin([self.test_fold, self.val_fold])]

    #     dfs = []
    #     for col in annotation_columns:
    #         if self.regression:
    #             aggregate_lambda = lambda x: x.mean()
    #         else:
    #             aggregate_lambda = lambda x: pd.Series.mode(x)[0]

    #         dfs.append(annotations.groupby("text_id")[col].apply(aggregate_lambda))

    #     annotations = pd.concat(dfs, axis=1).reset_index()
    #     annotations["annotator_id"] = 0
    #     annotations["split"] = "train"
    #     annotations["fold"] = annotations["text_id"].map(text_id_to_fold)

    #     self.annotations = pd.concat([annotations, val_test_annotations])