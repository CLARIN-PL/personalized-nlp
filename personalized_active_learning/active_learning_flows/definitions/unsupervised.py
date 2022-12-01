"""Plain and simple standard active learning flow."""
import copy
from typing import Optional, Dict, Any

from active_learning.callbacks.confidences import SaveConfidencesCallback
from personalized_active_learning.active_learning_flows.base import ActiveLearningFlowBase
from personalized_active_learning.algorithms.unsupervised_pretraining import IUnsupervisedPretrainer
from pytorch_lightning import loggers as pl_loggers

from personalized_active_learning.learning.training import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything


class UnsupervisedActiveLearningFlow(ActiveLearningFlowBase):
    def __init__(
        self, unsupervised_pretrainer: IUnsupervisedPretrainer, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._unsupervised_pretrainer = unsupervised_pretrainer

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
        # TODO: Will be changed
        # model_cls = type(self.model)
        # model = model_cls.load_state_dict(self.pretrained_model.state_dict())
        model = copy.deepcopy(self.pretrained_model)
        trainer = train_test(
            dataset,
            model=model,
            logger=logger,
            **train_kwargs,
        )

        if any(dataset.annotations.split == "none"):
            # gather confidence levels
            not_annotated_dataloader = dataset.custom_dataloader("none", shuffle=False)
            trainer.predict(dataloaders=not_annotated_dataloader, ckpt_path="best")

            self.confidences = confidences_callback.predict_outputs

        logger.experiment.finish()

    def experiment(
        self,
        max_amount: int,
        step_size: int,
        **kwargs,  # TODO: Leftover to not break compatibility with old code
    ):
        """Run AL.

        Args:
            max_amount: Maximum number of texts that should be annotated
                before AL is stopped.
            step_size: The number of texts that should be annotated in each AL cycle.

        """
        while self.annotated_amount < max_amount:
            # TODO: We will need to initialize model there
            self.pretrained_model = self._unsupervised_pretrainer.pretrain(
                dataset=self.dataset,
                model=self.model,
            )
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
