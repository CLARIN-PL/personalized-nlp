import numpy as np
import pandas as pd
from active_learning.callbacks.confidences import SaveConfidencesCallback
from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.learning.train import train_test
from personalized_nlp.utils.callbacks.personal_metrics import (
    PersonalizedMetricsCallback,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from profilehooks import profile

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
        reinforce_experiment_num: int = 25,
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
        self._train_confidences = None

        self._all_ys = []
        self._all_Xs = []

    @profile(entries=140, dirs=True)
    def experiment(self, max_amount: int, step_size: int, **kwargs):
        while self.annotated_amount < max_amount:
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated == 0:
                break
            self.add_annotations(step_size)
            regression_result = self._reinforce_iteration()

            mean_regression_r2 = regression_result["test_r2"].mean()
            additional_hparams = {"mean_regression_r2": mean_regression_r2}
            print("Final regression result")
            print(regression_result)
            self.train_model(additional_hparams=additional_hparams)

        if self.train_with_all_annotations:
            # Train at all annotations as baseline
            not_annotated = (self.datamodule.annotations.split == "none").sum()
            if not_annotated > 0:
                self.add_annotations(not_annotated)
                self.train_model(additional_hparams={})

    def _reinforce_iteration(self) -> dict:
        if self.reinforce_experiment_num == 0:
            return {}

        # result for full training dataset
        full_sample_result = self._reinforce_train_test()

        undersampled_indexes = []
        regression_result = {}

        for _ in range(self.reinforce_experiment_num):
            # subsample training dataset
            undersampled_indexes.append(self._subsample_train_annotations())

            # results for subsampled training datasets
            undersampled_f1_result = self._reinforce_train_test()

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

            undersampled_metric = metrics

            # revert subsampling training dataset
            self._revert_subsample()
            regression_result = self._improve_linear_regression(
                full_sample_result, undersampled_f1_result, undersampled_metric
            )

        return regression_result

    def _subsample_train_annotations(self) -> np.array:
        annotations = self.datamodule.annotations
        texts = self.datamodule.data

        annotated = annotations.loc[annotations["split"].isin(["train"])]
        not_annotated = annotations.loc[
            annotations["split"] == "none", ["text_id", "annotator_id"]
        ]

        not_annotated["original_index"] = not_annotated.index.values
        annotated["original_index"] = annotated.index.values
        selected = self.text_selector.select_annotations(
            texts,
            annotated,
            annotated,
            self._train_confidences,
            self.reinforce_subsample_num,
        )

        subsampled_idx = selected["original_index"].tolist()

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

        # metrics_callback = PersonalizedMetricsCallback()
        confidences_callback = SaveConfidencesCallback()
        # train_kwargs["custom_callbacks"] = [metrics_callback, confidences_callback]
        train_kwargs["custom_callbacks"] = [confidences_callback]

        print("Split value counts")
        print(datamodule.annotations.split.value_counts())

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

        annotated_dataloader = datamodule.custom_dataloader("train", shuffle=False)
        trainer = trainer_output["trainer"]
        trainer.predict(dataloaders=annotated_dataloader)

        self._train_confidences = confidences_callback.predict_outputs

        personal_f1 = -trainer_output["train_metrics"]["valid_loss"]

        return personal_f1.cpu().numpy()

    def _improve_linear_regression(
        self,
        full_sample_result: float,
        undersampled_f1_results: float,
        undersampled_metrics: np.ndarray,
    ):
        print("Learning regression")
        print("Full sample result:", full_sample_result)
        print("Undersampled F1 results", undersampled_f1_results)

        # calculate text selector metrics for undersampled annotations
        num_metrics = undersampled_metrics.shape[0]
        ys = [full_sample_result - undersampled_f1_results] * num_metrics
        self._all_ys.append(ys)
        self._all_Xs.append(undersampled_metrics)

        y = np.hstack(self._all_ys)
        X = np.vstack(self._all_Xs)

        df = pd.DataFrame(X)
        df["y"] = y
        filename = f"{y.shape[0]}.csv"
        df.to_csv(
            f"/home/mgruza/repos/personalized-nlp/active_learning/experiments/reinforce/{filename}",
            index=False,
        )

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
