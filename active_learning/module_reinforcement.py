from typing import List

import numpy as np
from active_learning.callbacks.confidences import SaveConfidencesCallback
from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.learning.train import train_test
from personalized_nlp.utils.callbacks.personal_metrics import (
    PersonalizedMetricsCallback,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

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
        reinforce_experiment_num: int = 50,
        reinforce_subsample_num: int = 200,
        **kwargs,
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
        self.reinforce_confidences = None

    def experiment(self, max_amount: int, step_size: int, **kwargs):
        while self.annotated_amount < max_amount:
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated == 0:
                break
            self.add_annotations(step_size)
            regression_result = self._reinforce_iteration()

            mean_regression_r2 = regression_result["test_r2"].mean()
            additional_hparams = {"mean_regression_r2": mean_regression_r2}
            self.train_model(additional_hparams=additional_hparams)

        if self.train_with_all_annotations:
            # Train at all annotations as baseline
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated > 0:
                self.add_annotations(not_annotated)
                self.train_model()

    def _reinforce_iteration(self) -> dict:
        if self.reinforce_experiment_num == 0:
            return {}

        # result for full training dataset
        full_sample_result = self._reinforce_train_test()

        undersampled_f1_results = []
        undersampled_indexes = []
        undersampled_metrics = []

        for _ in range(self.reinforce_experiment_num):
            # subsample training dataset
            undersampled_indexes.append(self._subsample_train_annotations())

            # results for subsampled training datasets
            undersampled_f1_results.append(self._reinforce_train_test())

            confidences = self.reinforce_confidences
            annotations = self.datamodule.annotations
            texts = self.datamodule.data

            annotated = annotations.loc[annotations["split"].isin(["train"])]
            subsampled = annotations.loc[
                annotations["split"] == "subsampled_for_reinforce",
                ["text_id", "annotator_id"],
            ]
            metrics = self.text_selector.get_metrics(
                texts, annotated, subsampled, confidences
            )

            undersampled_metrics.append(metrics)

            # revert subsampling training dataset
            self._revert_subsample()

        return self._improve_linear_regression(
            full_sample_result, undersampled_f1_results, undersampled_metrics
        )

    def _subsample_train_annotations(self) -> np.array:
        annotations = self.datamodule.annotations
        train_annotations = annotations.loc[annotations.split == "train"]
        train_annotations_idx = train_annotations.index.tolist()

        subsampled_idx = np.random.choice(
            train_annotations_idx, size=self.reinforce_subsample_num
        )

        annotations.loc[subsampled_idx, "split"] = "subsampled_for_reinforce"

        return subsampled_idx

    def _revert_subsample(self) -> None:
        annotations = self.datamodule.annotations
        subsampled_idx = annotations["split"] == "subsampled_for_reinforce"
        annotations.loc[subsampled_idx, "split"] = "train"

    def _reinforce_train_test(self) -> float:
        datamodule = self.datamodule
        model_kwargs = dict(self.model_kwargs)
        train_kwargs = dict(self.train_kwargs)

        metrics_callback = PersonalizedMetricsCallback()
        confidences_callback = SaveConfidencesCallback()
        train_kwargs["custom_callbacks"] = [metrics_callback, confidences_callback]

        trainer_output = train_test(
            datamodule, model_kwargs=model_kwargs, **train_kwargs, advanced_output=True
        )

        if any(datamodule.annotations.split == "subsampled_for_reinforce"):
            # gather confidence levels
            not_annotated_dataloader = datamodule.custom_dataloader(
                "subsampled_for_reinforce", shuffle=False
            )
            trainer = trainer_output["trainer"]
            trainer.predict(dataloaders=not_annotated_dataloader)

            self.reinforce_confidences = confidences_callback.predict_outputs

        personal_f1 = trainer_output["train_metrics"]["valid_personal_macro_f1"]

        return personal_f1.cpu().numpy()

    def _improve_linear_regression(
        self,
        full_sample_result: float,
        undersampled_f1_results: List[float],
        undersampled_metrics: List[np.ndarray],
    ):
        print("Learning regression")
        print("Full sample result:", full_sample_result)
        print("Undersampled F1 results", undersampled_f1_results)

        # calculate text selector metrics for undersampled annotations
        y_list = []
        for f1_result, metrics in zip(undersampled_f1_results, undersampled_metrics):
            ys = [full_sample_result - f1_result] * metrics.shape[0]
            y_list.append(ys)

        y = np.hstack(y_list)
        X = np.vstack(undersampled_metrics)

        # train linear regression to predict model f1 metrics
        cv_results = self._test_regression(X, y)
        reg = LinearRegression().fit(X, y)

        # update self.text_selector with new linear model
        self.text_selector.set_new_model(reg)

        return cv_results

    def _test_regression(self, X, y):
        reg = LinearRegression()
        cv_results = cross_validate(
            reg,
            X,
            y,
            cv=5,
            scoring=("r2", "neg_mean_squared_error"),
        )

        print(f"Reinforcement regression scoring: {cv_results}")

        return cv_results
