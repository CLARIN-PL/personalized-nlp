import numpy as np
import pandas as pd
import wandb

from personalized_nlp.learning.classifier import Classifier
from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.learning.train import train_test
from personalized_nlp.utils import seed_everything
from pytorch_lightning import loggers as pl_loggers
from settings import LOGS_DIR

from active_learning.algorithms.base import TextSelectorBase
from active_learning.callbacks.confidences import SaveConfidencesCallback
from personalized_nlp.models import models as models_dict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Final, Optional

from .unsupervised import IUnsupervisedPretrained


class ActiveLearningModule:
    """High level class defining how active learning algorithm looks like."""

    RANDOM_STATE: Final[int] = 24

    @property
    def annotated_amount(self) -> int:
        """Number of already annotated samples."""
        return self.datamodule.annotations.split.isin(["train"]).sum()

    def __init__(
        self,
        datamodule: BaseDataModule,
        text_selector: TextSelectorBase,
        datamodule_kwargs: dict,
        model_kwargs: dict,
        train_kwargs: dict,
        wandb_project_name: str,
        validation_ratio: float = 0.2,
        train_with_all_annotations: bool = True,
        stratify_by_user: bool = False,
        # TODO: Can be changed to optional class UnsupervisedPretrained to simplify ActiveLearningModule
        unsupervised_pretrainer: Optional[IUnsupervisedPretrained] = None,
        **kwargs
    ) -> None:
        """Initialize class.

        Args:
            datamodule: Data pool for active learning.
            text_selector: Selects which data should be annotated on each AL cycle.
            datamodule_kwargs: Keyword arguments for `datamodule`.
            model_kwargs: Model keyword arguments.
            train_kwargs: Model training keyword arguments.
            wandb_project_name: Project name in service Weights & Biases.
            validation_ratio: IT LOOKS LIKE IT IS NOT USED.
            train_with_all_annotations: Whether model should be train on all data.
            stratify_by_user: Whether data should be stratified by user. TODO: What it means?
            unsupervised_pretrainer: If provided applies used to supplement AL with unsupervised pretraining.
            kwargs: NOT USED.

        """
        self.datamodule = datamodule

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.datamodule_kwargs = datamodule_kwargs

        self.wandb_project_name = wandb_project_name
        self.validation_ratio = validation_ratio
        self.confidences = None
        self.train_with_all_annotations = train_with_all_annotations
        self.stratify_by_user = stratify_by_user
        self.unsupervised_pretrainer = unsupervised_pretrainer
        annotations = datamodule.annotations
        annotations.loc[annotations.split.isin(["train"]), "split"] = "none"

        self.text_selector = text_selector

    def add_annotations(self, amount: int):
        """Annotate texts.

        Args:
            amount: Number of texts that should be annotated.

        """
        annotations = self.datamodule.annotations
        texts = self.datamodule.data

        annotated = annotations.loc[annotations["split"].isin(["train"])].copy()
        not_annotated = annotations.loc[
            annotations["split"] == "none", ["text_id", "annotator_id"]
        ].copy()

        not_annotated["original_index"] = not_annotated.index.values
        annotated["original_index"] = annotated.index.values

        selected = self.text_selector.select_annotations(
            texts, annotated, not_annotated, self.confidences, amount
        )

        if not self.stratify_by_user:
            selected = selected.iloc[:amount]

        self._assign_train_val_splits(annotated, selected)

    def _assign_train_val_splits(
        self, old_annotations, selected_annotations: pd.DataFrame
    ):
        """Mark selected annotated data as training data.

        Args:
            old_annotations: NOT USED.
            selected_annotations: Annotations that should be used as training data.

        """
        annotations = self.datamodule.annotations
        annotations.loc[selected_annotations.index, "split"] = "train"

    def _perform_unsupervised_pretraining(self):
        """Perform unsupervised pre-training.

        As defined in paper Rethinking deep active learning:Using unlabeled data at model training.

        """
        # TODO: Verify
        output_dim = sum(self.datamodule.class_dims)
        # K-Means
        # Ugly since clients needs to know the structure of dict
        model_type = self.train_kwargs["model_type"]
        model_cls = models_dict[model_type]
        initial_model = model_cls(
            output_dim=output_dim,
            text_embedding_dim=self.datamodule.text_embedding_dim,
            annotator_num=self.datamodule.annotators_number,
            bias_vector_length=len(self.datamodule.class_dims),
            **self.model_kwargs,
        )
        # UGLY!
        # Value must be identical as provided to train_test_split
        lr = self.train_kwargs["lr"] if "lr" in self.train_kwargs else 1e-2
        initial_model = Classifier(
            model=initial_model,
            lr=lr,
            class_dims=self.datamodule.class_dims,
            class_names=self.datamodule.annotation_columns,
        )
        trainer = self.unsupervised_pretrainer.pretrain(
            data=self.datamodule.data,
            text_embeddings=self.datamodule.text_embeddings,
            model=initial_model,
            random_state=self.RANDOM_STATE,
        )
        self._unsupervised_model = trainer.model
        # TODO: Consider
        # self._unsupervised_model = Classifier.load_from_checkpoint(
        #     trainer.checkpoint,
        #     lr=lr,
        #     class_dims=self.datamodule.class_dims,
        #     class_names=self.datamodule.annotation_columns,
        # )

    def train_model(self):
        """Train model."""

        datamodule = self.datamodule
        # TODO: Do we need that dict()?
        datamodule_kwargs = dict(self.datamodule_kwargs)
        model_kwargs = dict(self.model_kwargs)
        train_kwargs = dict(self.train_kwargs)

        confidences_callback = SaveConfidencesCallback()
        if "custom_callbacks" in train_kwargs:
            train_kwargs["custom_callbacks"].append(confidences_callback)
        else:
            train_kwargs["custom_callbacks"] = [confidences_callback]

        annotations = self.datamodule.annotations
        annotated_annotations = annotations[annotations.split.isin(["train"])]
        annotated_texts_number = annotated_annotations.text_id.nunique()
        annotator_number = annotated_annotations.annotator_id.nunique()

        mean_positive = (
            annotated_annotations.loc[:, self.datamodule.annotation_columns]
            .mean(axis=0)
            .values
        )
        median_annotations_per_user = (
            annotated_annotations["annotator_id"].value_counts().median()
        )

        hparams = {
            "dataset": type(datamodule).__name__,
            "annotation_amount": self.annotated_amount,
            "text_selector": type(self.text_selector).__name__,
            "unique_texts_number": annotated_texts_number,
            "unique_annotator_number": annotator_number,
            "mean_positive": mean_positive,
            "median_annotations_per_user": median_annotations_per_user,
            "stratify_by_user": self.stratify_by_user,
            "use_unsupervised_pretraining": self.unsupervised_pretrainer is not None,
            **datamodule_kwargs,
            **model_kwargs,
            **train_kwargs,
        }
        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            config=hparams,
            project=self.wandb_project_name,
            log_model=False,
        )
        if self.unsupervised_pretrainer is not None:
            train_kwargs["pretrained_model"] = self._unsupervised_model

        seed_everything(self.RANDOM_STATE)
        trainer = train_test(
            datamodule,
            model_kwargs=model_kwargs,
            logger=logger,
            **train_kwargs,
        )

        logger.experiment.finish()

        if any(datamodule.annotations.split == "none"):
            # gather confidence levels
            not_annotated_dataloader = datamodule.custom_dataloader(
                "none", shuffle=False
            )
            trainer.predict(dataloaders=not_annotated_dataloader)

            self.confidences = confidences_callback.predict_outputs

    def experiment(self, max_amount: int, step_size: int, **kwargs):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated before AL is stopped.
            step_size: The number of texts that should be annotated in each AL cycle.
            **kwargs:

        """
        if self.unsupervised_pretrainer is not None:
            self._perform_unsupervised_pretraining()
        while self.annotated_amount < max_amount:
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated == 0:
                break
            self.add_annotations(step_size)
            self.train_model()

        if self.train_with_all_annotations:
            # Train at all annotations as baseline
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated > 0:
                self.add_annotations(not_annotated)
                self.train_model()
