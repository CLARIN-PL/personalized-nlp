from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.learning.train import train_test

import numpy as np

from active_learning.algorithms.base import TextSelectorBase
from active_learning.module import ActiveLearningModule


class ActiveReinforcementLearningModule(ActiveLearningModule):
    @property
    def annotated_amount(self) -> int:
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
        reinforce_experiment_num: int = 0,
        reinforce_subsample_num: int = 1,
        **kwargs
    ) -> None:

        super().__init__(
            datamodule,
            text_selector,
            datamodule_kwargs,
            model_kwargs,
            train_kwargs,
            wandb_project_name,
            validation_ratio,
            train_with_all_annotations,
            stratify_by_user,
        )
        self.reinforce_experiment_num = reinforce_experiment_num
        self.reinforce_subsample_num = reinforce_subsample_num

    def experiment(self, max_amount: int, step_size: int, **kwargs):
        while self.annotated_amount < max_amount:
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated == 0:
                break
            self.add_annotations(step_size)
            self._reinforce()
            self.train_model()

        if self.train_with_all_annotations:
            # Train at all annotations as baseline
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated > 0:
                self.add_annotations(not_annotated)
                self.train_model()

    def _reinforce(self):
        if self.reinforce_experiment_num == 0:
            return

        full_sample_result = self._reinforce_train_test()

        undersampled_results = []
        undersampled_indexes = []

        for _ in range(self.reinforce_experiment_num):
            undersampled_indexes.append(self._subsample_train_annotations())

            undersampled_results.append(self._reinforce_train_test())
            self._revert_subsample()

        self._improve_

    def _subsample_train_annotations(self):
        annotations = self.datamodule.annotations
        train_annotations = annotations.loc[annotations.split == "train"]
        train_annotations_idx = train_annotations.index.tolist()

        subsampled_idx = np.random.choice(
            train_annotations_idx, size=self.reinforce_subsample_num
        )

        annotations.loc[subsampled_idx, "split"] = "subsampled_for_reinforce"

    def _revert_subsample(self):
        annotations = self.datamodule.annotations
        subsampled_idx = annotations["split"] == "subsampled_for_reinforce"
        annotations.loc[subsampled_idx, "split"] = "train"

    def _reinforce_train_test(self):
        datamodule = self.datamodule
        model_kwargs = dict(self.model_kwargs)
        train_kwargs = dict(self.train_kwargs)

        trainer = train_test(
            datamodule,
            model_kwargs=model_kwargs,
            **train_kwargs,
        )

        personal_f1 = 0  # wyciągnąć personal f1 z trainer
        return personal_f1
