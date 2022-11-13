import abc
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import Dataset, BatchSampler, SubsetRandomSampler

from settings import LOGS_DIR, CHECKPOINTS_DIR


class IUnsupervisedPretrained(abc.ABC):
    """Perform unsupervised pre-training.

    As defined in paper Rethinking deep active learning:Using unlabeled data at model training.

    """

    @abc.abstractmethod
    def pretrain(
        self,
        data: pd.DataFrame,
        text_embeddings: torch.tensor,
        model: nn.Module,
        random_state: Optional[int] = None,
    ) -> pl.Trainer:
        """

        Args:
            data: Dataframe containing texts, etc. as present in DatamoduleBase.
            text_embeddings: Text embeddings.
            model: Model that will be pretrained.
            random_state: Random state.

        Returns:
            Trainer used to train model.
        """
        raise NotImplementedError


class KmeansDataset(Dataset):
    """Simple dataset to be used during K-Means pretraining."""

    def __init__(self, data: pd.DataFrame, y: np.ndarray, text_embeddings: torch.tensor):
        super(Dataset, self).__init__()
        self.data = data
        self.y = y
        self.text_embeddings = text_embeddings

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_data = {}
        # TODO: to use more advanced models we need there the same type of output as BatchIndexedDataset
        batch_data["embeddings"] = self.text_embeddings
        return batch_data, self.y[index]


class KmeansDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: pd.DataFrame,
        text_embeddings: torch.tensor,
        num_clusters: int,
        batch_size: int,
        random_seed: Optional[int] = None,
    ):
        super().__init__()
        self.data = data
        self.text_embeddings = text_embeddings
        # RandomState to receive different results each time K-Means is executed
        self.random_state = np.random.RandomState(seed=random_seed)
        self.num_clusters = num_clusters
        self.batch_size = batch_size

    def _run_kmeans(
        self,
        features: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        # We do not standardize data since this should be done before data are passed to KmeansDataModule
        return KMeans(
            n_clusters=num_clusters,
            random_state=self.random_state,
        ).fit_predict(features)

    def setup(self, stage: Optional[str] = None):
        # Original code uses only training set
        y_train = self._run_kmeans(features=self.text_embeddings.numpy(), num_clusters=self.num_clusters)
        # Same approach as in original paper
        sampler = SubsetRandomSampler(list(range(len(y_train))))
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=True
        )
        # TODO: Consider splitting data into train & test
        train_ds = KmeansDataset(data=self.data, y=y_train, text_embeddings=self.text_embeddings)
        self.train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
        )

    def train_dataloader(self):
        return self.train_dl


class KmeansScheduler(pl.Callback):
    def on_epoch_start(self, trainer, model):
        # Run K-Means
        trainer.datamodule.setup()


class KMeansUnsupervisedPretrainer(IUnsupervisedPretrained):
    """Pretrainer based on K-Means algorithm.

    [IMPORTANT]: K-Means is run every training epoch as in original paper!

    """

    def __init__(
        self,
        use_cuda: bool,
        num_classes: int,
        num_epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        self._num_epochs = num_epochs
        self._use_cuda = use_cuda
        self._batch_size = batch_size
        self._num_classes = num_classes

    def pretrain(
            self,
            data: pd.DataFrame,
            text_embeddings: torch.tensor,
            model: nn.Module,
            random_state: Optional[int] = None,
    ) -> pl.Trainer:
        """

        Args:
            data: Dataframe containing texts, etc. as present in DatamoduleBase.
            text_embeddings: Text embeddings.
            model: Model that will be pretrained.
            random_state: Random state.

        Returns:
            Trainer used to train model.
        """

        progressbar_checkpoint = TQDMProgressBar(refresh_rate=20)
        kmeans_scheduler = KmeansScheduler()
        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            log_model=False,
        )
        checkpoint_dir = CHECKPOINTS_DIR / "kmeans_pretraining"
        # TODO: Consider splitting data into train & validation
        #  and load model based on best val loss
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            monitor="train_loss",
            mode="min",
        )
        callbacks = [progressbar_checkpoint, kmeans_scheduler, checkpoint_callback]
        trainer = pl.Trainer(
            gpus=1 if self._use_cuda else 0,
            max_epochs=self._num_epochs,
            logger=logger,
            callbacks=callbacks,
        )
        datamodule = KmeansDataModule(
            data, text_embeddings, num_clusters=self._num_classes, batch_size=self._batch_size
        )
        trainer.fit(model, datamodule=datamodule)
        return trainer
