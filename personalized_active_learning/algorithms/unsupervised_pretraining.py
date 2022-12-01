import abc
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from personalized_active_learning.datasets import BaseDataset
from personalized_active_learning.datasets.types import TextFeaturesBatchDataset
from personalized_active_learning.models import IModel


class IUnsupervisedPretrainer(abc.ABC):
    """Perform unsupervised pre-training.

    As defined in paper "Rethinking deep active learning:
    Using unlabeled data at model training".

    """

    @abc.abstractmethod
    def pretrain(
        self,
        dataset: BaseDataset,
        model: IModel,
        seed: Optional[int] = None,
    ) -> pl.Trainer:
        """

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be pretrained.
            seed: Random seed.

        Returns:
            Trainer used to train model. Can be used to load model weights.

        """
        raise NotImplementedError


class _KmeansDataModule(pl.LightningDataModule):
    def __init__(
        self,
        text_ids: np.ndarray,
        annotator_ids: np.ndarray,
        annotator_biases: np.ndarray,
        raw_texts: np.ndarray,
        text_embeddings: torch.tensor,
        num_clusters: int,
        batch_size: int,
        random_seed: Optional[int] = None,
        use_text_ids_during_clustering: bool = True,
        use_annotator_ids_during_clustering: bool = True,
        num_workers: int = 4,
    ):
        """Initialize object.

        Args:
            text_ids: The texts' ids.
            annotator_ids: The annotators' ids.
            annotator_biases: The annotators' biases.
            raw_texts: The raw texts.
            text_embeddings: The text embeddings.
            num_clusters: The number of clusters that will be used.
            batch_size: The size of generated batches.
            random_seed: Random seed.
            use_text_ids_during_clustering: Whether to use text' ids during clustering.
            use_annotator_ids_during_clustering: : Whether to use annotators' ids
                during clustering.
            num_workers: Pytorch number of workers.

        """
        super().__init__()
        self.text_ids = text_ids
        self.annotator_ids = annotator_ids
        self.annotator_biases = annotator_biases
        self.text_embeddings = text_embeddings
        self.raw_texts = raw_texts
        # RandomState to receive different results each time K-Means is executed
        self.random_state = np.random.RandomState(seed=random_seed)
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.num_workers = num_workers
        kmeans_features = self.text_embeddings.detach().cpu().numpy()
        if use_text_ids_during_clustering:
            kmeans_features = np.column_stack([text_ids, kmeans_features])
        if use_annotator_ids_during_clustering:
            kmeans_features = np.column_stack([annotator_ids, kmeans_features])
        kmeans_features = StandardScaler().fit_transform(kmeans_features)
        self.kmeans_features = kmeans_features

    def _run_kmeans(
        self,
    ) -> np.ndarray:
        return KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
        ).fit_predict(self.kmeans_features)

    def reinitialize_pseudo_labels(self):
        """Reinitialize K-Means labels."""

        pseudo_labels = self._run_kmeans()
        dataset = TextFeaturesBatchDataset(
            text_ids=self.text_ids,
            annotator_ids=self.annotator_ids,
            embeddings=self.text_embeddings,
            raw_texts=self.raw_texts,
            annotator_biases=self.annotator_biases,
            y=pseudo_labels,
        )
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(
            sampler=sampler,
            batch_size=self.batch_size,
            drop_last=False,
        )
        self.train_dl = DataLoader(
            dataset,
            sampler=batch_sampler,
            batch_size=None,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        """Returns dataloader for training part of the dataset.

        In our case training set contains all data, split should be performed earlier.

        Returns:
            DataLoader: training dataloader for the dataset.
        """
        return self.train_dl


class _KmeansScheduler(pl.Callback):
    """Callback responsible of K-Means clusters reinitialization on each epoch start."""

    def on_epoch_start(self, trainer: pl.Trainer, model: IModel):
        """Run KMeans.

        Args:
            trainer: The trainer.
                IMPORTANT: Must contains datamodule of type _KmeansDataModule.
            model: Model used during training.

        """
        trainer.datamodule.reinitialize_pseudo_labels()
