# TODO: REFACTOR!!!!
import os
import warnings
from itertools import product

from pytorch_lightning.utilities.warnings import PossibleUserWarning

import active_learning.algorithms as algorithms
from personalized_active_learning.active_learning_flows import (
    StandardActiveLearningFlow,
)
from personalized_active_learning.datamodules import UnhealthyDataModule
from personalized_active_learning.datamodules.base import SplitMode
from personalized_active_learning.embeddings import EmbeddingsCreator
from personalized_active_learning.embeddings.personalised import (
    MultipleUserIdsEmbeddings,
    UserIdEmbeddings
)
from personalized_active_learning.metrics.personal_metrics import (
    PersonalizedMetricsCallback,
)
from personalized_active_learning.models import PersonalizedBaseline
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from settings import DATA_DIR

# False positive https://github.com/Lightning-AI/lightning/issues/11856
warnings.filterwarnings("ignore", category=PossibleUserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"


if __name__ == "__main__":
    wandb_project_name = "PNW_AL_Unhealthy_embeddings_debug"
    wandb_entity_name = None  # None if you don't want to use entity
    datamodule_cls = UnhealthyDataModule
    model_cls = PersonalizedBaseline
    use_cuda = True
    activelearning_kwargs_list = product_kwargs(
        {
            "text_selector_cls": [
                algorithms.RandomSelector,
                algorithms.BalancedConfidenceSelector,
                # algorithms.BalancedClassesPerUserSelector,
                # algorithms.BalancedClassesPerTextSelector,
                # algorithms.ConfidenceSelector,
                # algorithms.MaxPositiveClassSelector,
                # algorithms.ConfidenceAllDimsSelector,
                # algorithms.Confidencev2Selector,
                # algorithms.RandomImprovedSelector,
            ],
            "max_amount": [50_000],
            "step_size": [2_000],
            "amount_per_user": [2**8],
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
            "folds_num": [10],
            "subset_ratio": [1],
            "batch_size": [32],
            "test_fold_index": [0],
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "min_annotations_per_user_in_fold": [None],
            "personalized_embeddings_cls": [
                MultipleUserIdsEmbeddings,
                UserIdEmbeddings,
                None
            ]
        }
    )
    model_kwargs_list = product_kwargs(
        {
            "hidden_sizes": [
                [400, 800, 1600, 800, 400],
            ],  # used by MLP
            "dropout": [0.2],  # used by MLP
        }
    )
    trainer_kwargs_list = product_kwargs(
        {
            "epochs": [1],
            "lr": [0.001],
            "use_cuda": [use_cuda],
            "monitor_metric": ["valid_loss"],
            "monitor_mode": ["min"],
        }
    )
    active_learning_flows = [
        {
            "flow_cls": StandardActiveLearningFlow,
            "extra_kwargs": {},
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
        embedding_dim = datamodule_kwargs[
            "embeddings_creator"
        ].text_embedding_dim
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
            model_embedding_dim=embedding_dim,
            model_hidden_dims=model_kwargs["hidden_sizes"],
            model_dropout=model_kwargs["dropout"],
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
