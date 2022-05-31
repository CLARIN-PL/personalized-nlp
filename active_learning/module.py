import numpy as np
from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.learning.train import train_test
from personalized_nlp.settings import LOGS_DIR
from pytorch_lightning import loggers as pl_loggers

from active_learning.algorithms.base import TextSelectorBase
from active_learning.callbacks.confidences import SaveConfidencesCallback


class ActiveLearningModule:
    @property
    def annotated_amount(self) -> int:
        return self.datamodule.annotations.split.isin(["train", "val"]).sum()

    def __init__(
        self,
        datamodule: BaseDataModule,
        text_selector: TextSelectorBase,
        datamodule_kwargs: dict,
        model_kwargs: dict,
        train_kwargs: dict,
        wandb_project_name: str,
        validation_ratio: float = 0.2,
        **kwargs
    ) -> None:

        self.datamodule = datamodule

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.datamodule_kwargs = datamodule_kwargs

        self.wandb_project_name = wandb_project_name
        self.validation_ratio = validation_ratio
        self.confidences = None

        annotations = datamodule.annotations
        annotations.loc[annotations.split.isin(["train", "val"]), "split"] = "none"

        self.text_selector = text_selector

    def add_annotations(self, amount):
        annotations = self.datamodule.annotations
        texts = self.datamodule.data

        annotated = annotations.loc[annotations["split"].isin(["train", "val"])]
        not_annotated = annotations.loc[
            annotations["split"] == "none", ["text_id", "annotator_id"]
        ]
        not_annotated["original_index"] = not_annotated.index.values

        selected = self.text_selector.select_annotations(
            texts, amount, annotated, not_annotated, self.confidences
        )

        assert len(selected.index) <= amount

        self._assign_train_val_splits(selected)

    def _assign_train_val_splits(self, selected_annotations):
        annotations = self.datamodule.annotations

        selected_texts = selected_annotations["text_id"].unique()
        np.random.shuffle(selected_texts)

        train_amount = int(len(selected_texts) * (1 - self.validation_ratio))

        train_texts_ids = selected_texts[:train_amount]
        val_texts_ids = selected_texts[train_amount:]

        train_annotations = selected_annotations[
            selected_annotations.text_id.isin(train_texts_ids)
        ]
        val_annotations = selected_annotations[
            selected_annotations.text_id.isin(val_texts_ids)
        ]

        annotations.loc[train_annotations["original_index"], "split"] = "train"
        annotations.loc[val_annotations["original_index"], "split"] = "val"

    def train_model(self):
        datamodule = self.datamodule
        datamodule_kwargs = dict(self.datamodule_kwargs)
        model_kwargs = dict(self.model_kwargs)
        train_kwargs = dict(self.train_kwargs)

        confidences_callback = SaveConfidencesCallback()
        if "custom_callbacks" in train_kwargs:
            train_kwargs["custom_callbacks"].append(confidences_callback)
        else:
            train_kwargs["custom_callbacks"] = [confidences_callback]

        hparams = {
            "dataset": type(datamodule).__name__,
            "annotation_amount": self.annotated_amount,
            "text_selector": type(self.text_selector).__name__,
            **datamodule_kwargs,
            **model_kwargs,
            **train_kwargs,
        }

        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            config=hparams,
            project=self.wandb_project_name,
            log_model=False
        )

        trainer = train_test(
            datamodule,
            model_kwargs=model_kwargs,
            logger=logger,
            **train_kwargs,
        )

        logger.experiment.finish()

        # gather confidence levels
        not_annotated_dataloader = datamodule.custom_dataloader("none", shuffle=False)
        trainer.predict(dataloaders=not_annotated_dataloader)

        self.confidences = confidences_callback.predict_outputs

    def experiment(self, max_amount: int, step_size: int, **kwargs):
        while self.annotated_amount < max_amount:
            self.add_annotations(step_size)
            self.train_model()