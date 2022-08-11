from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.learning.train import train_test
from personalized_nlp.utils import seed_everything
from pytorch_lightning import loggers as pl_loggers
from settings import LOGS_DIR

from active_learning.algorithms.base import TextSelectorBase
from active_learning.callbacks.confidences import SaveConfidencesCallback


class ActiveLearningModule:

    @property
    def annotated_amount(self) -> int:
        return self.datamodule.annotations.split.isin(["train"]).sum()

    def __init__(self,
                 datamodule: BaseDataModule,
                 text_selector: TextSelectorBase,
                 datamodule_kwargs: dict,
                 model_kwargs: dict,
                 train_kwargs: dict,
                 wandb_project_name: str,
                 validation_ratio: float = 0.2,
                 train_with_all_annotations=True,
                 stratify_by_user=False,
                 **kwargs) -> None:

        self.datamodule = datamodule

        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.datamodule_kwargs = datamodule_kwargs

        self.wandb_project_name = wandb_project_name
        self.validation_ratio = validation_ratio
        self.confidences = None
        self.train_with_all_annotations = train_with_all_annotations
        self.stratify_by_user = stratify_by_user
        annotations = datamodule.annotations
        annotations.loc[annotations.split.isin(["train"]), "split"] = "none"

        self.text_selector = text_selector

    def add_annotations(self, amount):
        annotations = self.datamodule.annotations
        texts = self.datamodule.data

        annotated = annotations.loc[annotations["split"].isin(["train"])]
        not_annotated = annotations.loc[annotations["split"] == "none",
                                        ["text_id", "annotator_id"]]

        not_annotated["original_index"] = not_annotated.index.values
        annotated["original_index"] = annotated.index.values

        selected = self.text_selector.select_annotations(
            texts, annotated, not_annotated, self.confidences, amount)

        if not self.stratify_by_user:
            selected = selected.iloc[:amount]

        self._assign_train_val_splits(annotated, selected)

    def _assign_train_val_splits(self, old_annotations, selected_annotations):
        annotations = self.datamodule.annotations
        annotations.loc[selected_annotations.index, "split"] = "train"

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

        annotations = self.datamodule.annotations
        annotated_annotations = annotations[annotations.split.isin(["train"])]
        annotated_texts_number = annotated_annotations.text_id.nunique()
        annotator_number = annotated_annotations.annotator_id.nunique()

        mean_positive = (
            annotated_annotations.loc[:,
                                      self.datamodule.annotation_columns].mean(
                                          axis=0).values)
        median_annotations_per_user = (
            annotated_annotations["annotator_id"].value_counts().median())

        hparams = {
            "dataset": type(datamodule).__name__,
            "annotation_amount": self.annotated_amount,
            "text_selector": type(self.text_selector).__name__,
            "unique_texts_number": annotated_texts_number,
            "unique_annotator_number": annotator_number,
            "mean_positive": mean_positive,
            "median_annotations_per_user": median_annotations_per_user,
            "stratify_by_user": self.stratify_by_user,
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

        seed_everything(24)
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
                "none", shuffle=False)
            trainer.predict(dataloaders=not_annotated_dataloader)

            self.confidences = confidences_callback.predict_outputs

    def experiment(self, max_amount: int, step_size: int, **kwargs):
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
