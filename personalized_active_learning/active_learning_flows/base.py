from typing import Optional, List

import pandas as pd

from personalized_active_learning.datasets import BaseDataModule
from personalized_active_learning.learning.training import train_test
from personalized_active_learning.models import IModel
from personalized_nlp.utils import seed_everything
from active_learning.algorithms.base import TextSelectorBase
from active_learning.callbacks.confidences import SaveConfidencesCallback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Callback
from settings import LOGS_DIR
import abc


class ActiveLearningFlowBase(abc.ABC):
    @property
    def annotated_amount(self) -> int:
        """Number of already annotated samples."""
        return self.dataset.annotations.split.isin(["train"]).sum()

    def __init__(
        self,
        dataset: BaseDataModule,
        text_selector: TextSelectorBase,
        model_cls: IModel,
        wandb_project_name: str,
        logger_extra_metrics: dict,
        monitor_metric: str,
        monitor_mode: str,
        epochs: int,
        lr: float,
        use_cuda: bool,
        model_output_dim: int,
        model_embedding_dim: int,
        model_hidden_dims: Optional[List[int]] = None,
        custom_callbacks: Optional[List[Callback]] = None,
        stratify_by_user: bool = False,
    ) -> None:
        """Base class for AL algorithm.

        Class contains basic flow of algorithm with all common functions.
        In a standard scenario only `experiment` method is needed to be
        implemented.

        If you want to add a new parameter, you need to add it to one of the
        dictionaries below: global - `self.flow_params`; model - `self.model_params`;
        trainer - `trainer_params`.

        Args:
            dataset (BaseDataModule): A dataset which derives from the
                `personalized_active_learning.datasets.base` class.
            text_selector (TextSelectorBase): A text selector which derives from
                `active_learning.algorithms.base` class. Selects annotations from dataset.
            model_cls (IModel): A model defined in `personalized_active_learning.models`
                directory. Used for a main training. (Imorted definition, eg. `Baseline`).
            wandb_project_name (str): A name of the experiment.
            logger_extra_metrics (dict): Dictionary with extra metrics for logger.
                By default `datamodule_kwargs` should be passed
            monitor_metric (str): Monitor argument for `ModelCheckpoint`.
                https://keras.io/api/callbacks/model_checkpoint/
            monitor_mode (str): Mode argument for `ModelCheckpoint`
                  https://keras.io/api/callbacks/model_checkpoint/
            epochs (int): Number of training epochs. Used in `training` module.
            lr (float): Learning rate for a model. Used in `training` module.
            use_cuda (bool): Use cuda. Used in `training` module.
            model_output_dim (int): Output dimension used in `model_cls` initialization
            model_embedding_dim (int): Embeding dimension used in `model_cls`
                initialization.
            model_hidden_dims Optional[List[int]]: Hidden dimensions sizes used in
                `model_cls`initialization. Defaults to None.
            custom_callbacks (Optional[List[Callback]]): A list with custom callbacks for
                `Trainer`.
            `SaveConfidencesCallback`, `ModelCheckpoint` and `TQDMProgressBar` are
            always added. Defaults to None.
            stratify_by_user (bool, optional): Stratify data by user. Defaults to False.
        """

        self.confidences = None
        self.dataset = dataset
        self.logger_extra_metrics = logger_extra_metrics
        self.wandb_project_name = wandb_project_name
        self.stratify_by_user = stratify_by_user
        self.text_selector = text_selector
        self.model_cls = model_cls
        annotations = dataset.annotations

        annotations.loc[annotations.split.isin(["train"]), "split"] = "none"

        if not custom_callbacks:
            custom_callbacks = []

        self.confidences_callback = SaveConfidencesCallback()
        custom_callbacks.append(self.confidences_callback)

        # Arguments passed in train_model
        self.flow_params = {
            "datamodule": dataset,
        }

        self.model_params = {
            "output_dim": model_output_dim,
            "embeding_dim": model_embedding_dim,
            "hidden_dims": model_hidden_dims,  # Added for future
        }

        self.trainer_params = {
            "epochs": epochs,
            "lr": lr,
            "use_cuda": use_cuda,
            "monitor_metric": monitor_metric,
            "monitor_mode": monitor_mode,
            "custom_callbacks": custom_callbacks,
        }

    def add_annotations(self, amount: int):
        """Annotate texts.

        Args:
            amount: Number of texts that should be annotated.

        """
        annotations = self.dataset.annotations
        texts = self.dataset.data

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

        self._assign_train_val_splits(selected)

    def _assign_train_val_splits(self, selected_annotations: pd.DataFrame):
        """Mark selected annotated data as training data.

        Args:
            selected_annotations: Annotations that should be used as training data.

        """
        annotations = self.dataset.annotations
        annotations.loc[selected_annotations.index, "split"] = "train"

    def train_model(self) -> None:
        """Run single iteration of AL algorithm.

        Prepare logger, run training and save metrics. As a result `self.confidences`
        are updated.
        """

        dataset = self.dataset
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

        lparams = {
            "dataset": type(dataset).__name__,
            "annotation_amount": self.annotated_amount,
            "text_selector": type(self.text_selector).__name__,
            "unique_texts_number": annotated_texts_number,
            "unique_annotator_number": annotator_number,
            "mean_positive": mean_positive,
            "median_annotations_per_user": median_annotations_per_user,
            "stratify_by_user": self.stratify_by_user,
            **self.logger_extra_metrics,
            **self.trainer_params,
        }

        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            project=self.wandb_project_name,
            log_model=False,
        )
        logger.experiment.config.update(lparams)
        seed_everything()  # Model's weights initialization
        model = self.model_cls(
            **self.model_params,
        )
        trainer = train_test(
            model=model, logger=logger, **self.flow_params, **self.trainer_params
        )

        if any(dataset.annotations.split == "none"):
            # Gather confidence levels
            not_annotated_dataloader = dataset.custom_dataloader("none", shuffle=False)
            trainer.predict(dataloaders=not_annotated_dataloader, ckpt_path="best")
            self.confidences = self.confidences_callback.predict_outputs

        logger.experiment.finish()

    @abc.abstractmethod
    def experiment(self, max_amount: int, step_size: int):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated before AL is
                stopped.
            step_size: The number of texts that should be annotated in each AL cycle.

        """
