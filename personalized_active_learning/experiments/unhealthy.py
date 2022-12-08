# TODO: REFACTOR!!!!
import os
from itertools import product

from pytorch_lightning.utilities.warnings import PossibleUserWarning

import active_learning.algorithms as algorithms
from personalized_active_learning.active_learning_flows import (
    StandardActiveLearningFlow,
    UnsupervisedActiveLearningFlow,
)
from personalized_active_learning.algorithms import KmeansPretrainer
from personalized_active_learning.datasets import UnhealthyDataset
from personalized_active_learning.datasets.base import SplitMode
from personalized_active_learning.embeddings import EmbeddingsCreator
from personalized_active_learning.metrics.personal_metrics import (
    PersonalizedMetricsCallback,
)
from personalized_active_learning.models import Baseline

from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
import warnings
from settings import DATA_DIR

# False positive https://github.com/Lightning-AI/lightning/issues/11856
warnings.filterwarnings("ignore", category=PossibleUserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"


if __name__ == "__main__":
    wandb_project_name = "PNW_AL_Unhealthy_subset_20"
    wandb_entity_name = "be-active"  # None if you don't want to use entity
    datamodule_cls = UnhealthyDataset
    model_cls = Baseline
    use_cuda = True
    activelearning_kwargs_list = product_kwargs(
        {
            "text_selector_cls": [
                algorithms.RandomSelector,
                algorithms.BalancedConfidenceSelector,
                algorithms.BalancedClassesPerUserSelector,
                algorithms.BalancedClassesPerTextSelector,
                # algorithms.ConfidenceSelector,
                # algorithms.MaxPositiveClassSelector,
                # algorithms.ConfidenceAllDimsSelector,
                # algorithms.Confidencev2Selector,
                # algorithms.RandomImprovedSelector,
            ],
            "max_amount": [50_000],
            "step_size": [2_000],
            "amount_per_user": [2],
            "stratify_by_user": [True],
        }
    )
    datamodule_kwargs_list = product_kwargs(
        {
            "embeddings_creator": [
                EmbeddingsCreator(
                    directory=DATA_DIR / "unhealthy_conversations" / "embeddings",
                    embeddings_type="labse",
                    use_cuda=use_cuda,
                ),
            ],
            "past_annotations_limit": [None],
            "split_mode": [SplitMode.TEXTS],
            "folds_num": [5],
            "subset_ratio": [0.2],
            "batch_size": [3000],
            "test_fold_index": list(range(5)),  # This does cross-validation
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "min_annotations_per_user_in_fold": [20],
        }
    )
    model_kwargs_list = product_kwargs(
        {
            "embedding_dim": [50],
            "hidden_dim": [100],
        }
    )
    trainer_kwargs_list = product_kwargs(
        {
            "epochs": [20],
            "lr": [0.008],
            "use_cuda": [use_cuda],
            "monitor_metric": ["valid_loss"],
            "monitor_mode": ["max"],
        }
    )
    active_learning_flows = [
        {
            "flow_cls": StandardActiveLearningFlow,
            "extra_kwargs": {},
        },
        {
            "flow_cls": UnsupervisedActiveLearningFlow,
            "extra_kwargs": {
                "unsupervised_pretrainer": KmeansPretrainer(
                    num_clusters=10,
                    batch_size=32,
                    wandb_project_name=wandb_project_name,  # TODO: Not sure about that
                    number_of_epochs=9,
                ),
            },
        },
    ]
    for (
        activelearning_kwargs,
        datamodule_kwargs,
        model_kwargs,
        trainer_kwargs,
        active_learning_flow,
    ) in product(
        activelearning_kwargs_list,
        datamodule_kwargs_list,
        model_kwargs_list,
        trainer_kwargs_list,
        active_learning_flows,
    ):
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)
        class_dimensions = data_module.classes_dimensions

        text_selector_cls = activelearning_kwargs["text_selector_cls"]
        text_selector = text_selector_cls(
            class_dims=class_dimensions,
            annotation_columns=data_module.annotation_columns,
            amount_per_user=activelearning_kwargs["amount_per_user"],
        )
        activelearning_kwargs["text_selector"] = text_selector
        trainer_kwargs["custom_callbacks"] = [
            PersonalizedMetricsCallback(),
        ]

        module = active_learning_flow["flow_cls"](
            dataset=data_module,
            model_cls=model_cls,
            wandb_project_name=wandb_project_name,
            wandb_entity_name=wandb_entity_name,
            model_output_dim=sum(class_dimensions),
            model_embedding_dim=model_kwargs["embedding_dim"],
            text_selector=activelearning_kwargs["text_selector"],
            stratify_by_user=activelearning_kwargs["stratify_by_user"],
            logger_extra_metrics=dict(datamodule_kwargs),
            **trainer_kwargs,
            **active_learning_flow["extra_kwargs"],
        )

        module.experiment(
            max_amount=activelearning_kwargs["max_amount"],
            step_size=activelearning_kwargs["step_size"],
        )
