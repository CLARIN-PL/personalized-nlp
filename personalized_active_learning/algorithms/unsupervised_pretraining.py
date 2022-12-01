import abc
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torchmetrics import Accuracy, F1Score

from personalized_active_learning.datasets import BaseDataset
from personalized_active_learning.datasets.types import TextFeaturesBatchDataset
from personalized_active_learning.models import IModel
from settings import CHECKPOINTS_DIR
from personalized_nlp.utils import PrecisionClass, RecallClass, F1Class
from settings import LOGS_DIR


class IUnsupervisedPretrainer(abc.ABC):
    """Perform unsupervised pre-training."""

    @abc.abstractmethod
    def pretrain(
        self,
        dataset: BaseDataset,
        model: IModel,
    ) -> IModel:
        """Pretrain model.

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be pretrained.

        Returns:
            Pretrained model.

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
        # NOTE: Single embedding might be used multiple times during training
        # Personalization effect
        kmeans_features = self.text_embeddings[text_ids.tolist()].detach().cpu().numpy()
        if use_text_ids_during_clustering:
            kmeans_features = np.column_stack([text_ids, kmeans_features])
        if use_annotator_ids_during_clustering:
            kmeans_features = np.column_stack([annotator_ids, kmeans_features])
        kmeans_features = StandardScaler().fit_transform(kmeans_features)
        self.kmeans_features = kmeans_features
        # Initial pseudo labels
        self.reinitialize_pseudo_labels()

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

    def on_epoch_end(self, trainer: pl.Trainer, model: IModel):
        """Run KMeans.

        Args:
            trainer: The trainer.
                IMPORTANT: Must contains datamodule of type _KmeansDataModule.
            model: Model used during training.

        """
        trainer.datamodule.reinitialize_pseudo_labels()


# TODO: Docstrings & typing
class _KmeansClassifier(pl.LightningModule):
    """Custom classifier for KMeans training."""

    def __init__(
        self,
        model: IModel,
        number_of_classes: int,
        lr: float,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.lr = lr

        self.metric_types = ("accuracy", "precision", "recall", "f1", "macro_f1")
        task_name = "kmeans"

        metrics = {}

        for split in ["train"]:
            metrics[f"{split}_accuracy_{task_name}"] = Accuracy()
            for class_idx in range(number_of_classes):
                metrics[f"{split}_precision_{task_name}_{class_idx}"] = PrecisionClass(
                    num_classes=number_of_classes, average=None, class_idx=class_idx
                )
                metrics[f"{split}_recall_{task_name}_{class_idx}"] = RecallClass(
                    num_classes=number_of_classes, average=None, class_idx=class_idx
                )
                metrics[f"{split}_f1_{task_name}_{class_idx}"] = F1Class(
                    average="none", num_classes=number_of_classes, class_idx=class_idx
                )
                metrics[f"{split}_macro_f1_{task_name}"] = F1Score(
                    average="macro", num_classes=number_of_classes
                )

        self.metrics = nn.ModuleDict(metrics)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def step(output, y):
        return nn.CrossEntropyLoss()(output, y.long())

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        X, y = batch

        output = self.forward(X)
        loss = self.step(output=output, y=y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        preds = torch.argmax(output, dim=1)

        return {
            "loss": loss,
            "output": output,
            "preds": preds,
            "y": y,
            "X": X,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
        output = torch.softmax(output, dim=1)
        log_dict = {}
        for metric_type in self.metric_types:
            metric_key_prefix = f"{split}_{metric_type}_{self.task_name}"

            metric_keys = [
                k for k in self.metrics.keys() if k.startswith(metric_key_prefix)
            ]
            for metric_key in metric_keys:
                self.metrics[metric_key](output.float(), y.long())

                log_dict[metric_key] = self.metrics[metric_key]

            # Dunno why
            if split == "valid" and metric_type == "macro_f1":
                f1_macros = [
                    log_dict[metric_key].compute().cpu() for metric_key in metric_keys
                ]
                log_dict["valid_macro_f1_mean"] = np.mean(f1_macros)

        self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=True)


def _train_kmeans_classifier(
    datamodule: _KmeansDataModule,
    model: IModel,
    number_of_classes: int,
    wandb_project_name: str,
    epochs: int,
    lr: float,
    use_cuda: bool = False,
    monitor_metric: str = "train_loss",
    monitor_mode: str = "min",
):
    logger = pl_loggers.WandbLogger(
        save_dir=str(LOGS_DIR),
        project=wandb_project_name,
        log_model=False,
    )

    classifier = _KmeansClassifier(
        model=model, lr=lr, number_of_classes=number_of_classes
    )
    checkpoint_dir = CHECKPOINTS_DIR / logger.experiment.name / "kmeans"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, save_top_k=1, monitor=monitor_metric, mode=monitor_mode
    )
    progressbar_checkpoint = TQDMProgressBar(refresh_rate=20)

    # Safety check
    use_cuda = use_cuda and torch.cuda.is_available()

    callbacks = [
        checkpoint_callback,
        progressbar_checkpoint,
        _KmeansScheduler(),
    ]

    trainer = pl.Trainer(
        gpus=1 if use_cuda else 0,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(classifier, datamodule)
    return classifier.model


class KmeansPretrainer(IUnsupervisedPretrainer):
    """Perform unsupervised pre-training based on K-Means clustering.

    As defined in paper "Rethinking deep active learning:
    Using unlabeled data at model training".

    """

    def __init__(
        self,
        num_clusters: int,
        batch_size: int,
        wandb_project_name: str,
        random_seed: Optional[int] = None,
        use_text_ids_during_clustering: bool = True,
        use_annotator_ids_during_clustering: bool = True,
        num_workers: int = 4,
        lr: float = 1e-2,
        number_of_epochs: int = 6,
        use_cuda: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self._wandb_project_name = wandb_project_name
        self._num_clusters = num_clusters
        self._batch_size = batch_size
        self._random_seed = random_seed
        self._use_text_ids_during_clustering = use_text_ids_during_clustering
        self._use_annotator_ids_during_clustering = use_annotator_ids_during_clustering
        self._num_workers = num_workers
        self._lr = lr
        self._number_of_epochs = number_of_epochs
        self._use_cuda = use_cuda
        self._random_seed = seed

    def pretrain(
        self,
        dataset: BaseDataset,
        model: IModel,
    ) -> IModel:
        """Pretrain model.

        Args:
            dataset: Dataset containing data used for pretraining.
            model: Model that will be pretrained.

        Returns:
            Pretrained model.

        """
        # Create dataset
        annotations = dataset.annotations
        is_annotation_for_training = annotations.split == "none"
        annotations = annotations.loc[is_annotation_for_training]
        data, y = dataset.get_data_and_labels(annotations)
        text_ids = data["text_id"]
        annotator_ids = data["annotator_id"]
        kmeans_data_module = _KmeansDataModule(
            text_ids=text_ids.values,
            annotator_ids=annotator_ids.values,
            text_embeddings=dataset.text_embeddings,
            raw_texts=dataset.data["text"].values,
            annotator_biases=dataset.annotator_biases.values.astype(float),
            num_clusters=self._num_clusters,
            batch_size=self._batch_size,
            random_seed=self._random_seed,
            use_text_ids_during_clustering=self._use_text_ids_during_clustering,
            use_annotator_ids_during_clustering=self._use_annotator_ids_during_clustering,
            num_workers=self._num_workers,
        )
        # Swap model head
        original_model_head = model.head
        model.head = nn.Linear(
            in_features=original_model_head.in_features, out_features=self._num_clusters
        )
        # Train model
        model = _train_kmeans_classifier(
            model=model,
            datamodule=kmeans_data_module,
            number_of_classes=self._num_clusters,
            wandb_project_name=self._wandb_project_name,
            epochs=self._number_of_epochs,
            lr=self._lr,
            use_cuda=self._use_cuda,
        )
        # Swap back model head
        model.head = original_model_head
        return model
