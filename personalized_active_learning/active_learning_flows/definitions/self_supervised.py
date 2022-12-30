"""Plain and simple standard active learning flow."""
from typing import Optional, List

from personalized_active_learning.learning.training import train_test
from personalized_nlp.utils import seed_everything
from pytorch_lightning import loggers as pl_loggers
from settings import LOGS_DIR

from personalized_active_learning.active_learning_flows.base import ActiveLearningFlowBase
from personalized_active_learning.algorithms.label_propagation.interface import ILabelPropagator
from personalized_active_learning.algorithms.self_supervised_learning import ISelfSupervisedTrainer


class SelfSupervisedActiveLearningFlow(ActiveLearningFlowBase):
    def __init__(self,
                 label_propagator: ILabelPropagator,
                 unsupervised_pretrainer: ISelfSupervisedTrainer,
                 train_with_all_annotations: bool = True,
                 **kwargs
    ) -> None:
        """Initialize object.

        Args:
            train_with_all_annotations: Whether model should be additionally trained
                with all annotations as baseline.
            kwargs: Keywords arguments for `ActiveLearningFlowBase`.
        """
        super().__init__(**kwargs)
        self.label_propagator = label_propagator
        self.unsupervised_pretrainer = unsupervised_pretrainer
        self.train_with_all_annotations = train_with_all_annotations

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
            "subset_ratio": self.subset_ratio,
            "dataset_size": len(dataset.data),
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
            entity=self.wandb_entity_name,
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
            # Propagate labels
            labelled_data,  unlabelled_data, pseudo_labels = self.label_propagator.propagate_labels(dataset)
            # use self supervised learning
            # Gather confidence levels
            not_annotated_dataloader = dataset.custom_dataloader("none", shuffle=False)
            trainer.predict(dataloaders=not_annotated_dataloader, ckpt_path="best")
            self.confidences = self.confidences_callback.predict_outputs

        logger.experiment.finish()



    def experiment(
        self,
        max_amount: int,
        step_size: int,
    ):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated before
                AL is stopped.
            step_size: The number of texts that should be annotated in each AL cycle.

        """
        while self.annotated_amount < max_amount:
            not_annotated = (self.dataset.annotations.split == "none").sum()
            if not_annotated == 0:
                break
            self.add_annotations(step_size)
            self.train_model()

        if self.train_with_all_annotations:
            # Train at all annotations as baseline
            not_annotated = (self.dataset.annotations.split == "none").sum()
            if not_annotated > 0:
                self.add_annotations(not_annotated)
                self.train_model()
