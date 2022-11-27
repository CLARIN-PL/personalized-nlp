# TODO: Refactor!!!
from typing import Any, Dict, Optional

import pandas as pd

from personalized_active_learning.datasets import BaseDataset
from personalized_active_learning.learning.training import train_test
from personalized_active_learning.models import IModel
from personalized_nlp.utils import seed_everything
from pytorch_lightning import loggers as pl_loggers
from settings import LOGS_DIR
from active_learning.algorithms.base import TextSelectorBase
from active_learning.callbacks.confidences import SaveConfidencesCallback
import abc


class ActiveLearningFlowBase(abc.ABC):
    @property
    def annotated_amount(self) -> int:
        """Number of already annotated samples."""
        return self.dataset.annotations.split.isin(["train"]).sum()

    def __init__(
        self,
        dataset: BaseDataset,
        text_selector: TextSelectorBase,
        datamodule_kwargs: dict,  # TODO: Should be removed
        model: IModel,
        train_kwargs: dict,  # TODO: Should be changed to Trainer
        wandb_project_name: str,  # TODO: Pass already initialized logger
        train_with_all_annotations=True,
        stratify_by_user=False,
        **kwargs,  # TODO: Remove, leftover to not break compatibility with old code
    ) -> None:
        self.model = model
        self.dataset = dataset

        self.train_kwargs = train_kwargs
        self.datamodule_kwargs = datamodule_kwargs

        self.wandb_project_name = wandb_project_name
        self.confidences = None
        self.train_with_all_annotations = train_with_all_annotations
        self.stratify_by_user = stratify_by_user
        annotations = dataset.annotations
        annotations.loc[annotations.split.isin(["train"]), "split"] = "none"

        self.text_selector = text_selector

    def add_annotations(self, amount: int):
        """Annotate texts.

        Args:
            amount: Number of texts that should be annotated.

        """
        annotations = self.dataset.annotations
        texts = self.dataset.data

        annotated = annotations.loc[annotations["split"].isin(["train"])]
        not_annotated = annotations.loc[
            annotations["split"] == "none", ["text_id", "annotator_id"]
        ]

        not_annotated["original_index"] = not_annotated.index.values
        annotated["original_index"] = annotated.index.values

        selected = self.text_selector.select_annotations(
            texts, annotated, not_annotated, self.confidences, amount
        )

        if not self.stratify_by_user:
            selected = selected.iloc[:amount]

        self._assign_train_val_splits(selected)

    def _assign_train_val_splits(self, selected_annotations: pd.DataFrame):
        """Mark selected annotated data as training data.

        Args:
            selected_annotations: Annotations that should be used as training data.

        """
        annotations = self.dataset.annotations
        annotations.loc[selected_annotations.index, "split"] = "train"

    def train_model(self, additional_hparams: Optional[Dict[str, Any]] = None):
        dataset = self.dataset
        datamodule_kwargs = dict(self.datamodule_kwargs)
        train_kwargs = dict(self.train_kwargs)

        confidences_callback = SaveConfidencesCallback()
        if "custom_callbacks" in train_kwargs:
            train_kwargs["custom_callbacks"].append(confidences_callback)
        else:
            train_kwargs["custom_callbacks"] = [confidences_callback]

        annotations = self.dataset.annotations
        annotated_annotations = annotations[annotations.split.isin(["train"])]
        annotated_texts_number = annotated_annotations.text_id.nunique()
        annotator_number = annotated_annotations.annotator_id.nunique()

        mean_positive = (
            annotated_annotations.loc[:, self.dataset.annotation_columns]
            .mean(axis=0)
            .values
        )
        median_annotations_per_user = (
            annotated_annotations["annotator_id"].value_counts().median()
        )

        hparams = {
            "dataset": type(dataset).__name__,
            "annotation_amount": self.annotated_amount,
            "text_selector": type(self.text_selector).__name__,
            "unique_texts_number": annotated_texts_number,
            "unique_annotator_number": annotator_number,
            "mean_positive": mean_positive,
            "median_annotations_per_user": median_annotations_per_user,
            "stratify_by_user": self.stratify_by_user,
            **datamodule_kwargs,
            **train_kwargs,
        }

        if additional_hparams is not None:
            hparams.update(additional_hparams)

        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            config=hparams,
            project=self.wandb_project_name,
            log_model=False,
        )

        seed_everything(24)
        trainer = train_test(
            dataset,
            model=self.model,
            logger=logger,
            **train_kwargs,
        )

        if any(dataset.annotations.split == "none"):
            # gather confidence levels
            not_annotated_dataloader = dataset.custom_dataloader("none", shuffle=False)
            trainer.predict(dataloaders=not_annotated_dataloader)

            self.confidences = confidences_callback.predict_outputs

        logger.experiment.finish()

    @abc.abstractmethod
    def experiment(self, max_amount: int, step_size: int):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated before AL is stopped.
            step_size: The number of texts that should be annotated in each AL cycle.

        """
