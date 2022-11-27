# TODO: REFACTOR!!!!
import os
from itertools import product

from pytorch_lightning.utilities.warnings import PossibleUserWarning

import active_learning.algorithms as algorithms
from personalized_active_learning.active_learning_flows import StandardActiveLearningFlow
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "PNW_AL_Unhealthy"
    datamodule_cls = UnhealthyDataset
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
            "batch_size": [3000],
            "test_fold_index": list(range(5)),  # This does cross-validation
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "min_annotations_per_user_in_fold": [20],
        }
    )
    # TODO: Needs to be changed
    model_kwargs_list = product_kwargs(
        {
            "embedding_dim": [50],
            "dp_emb": [0.25],
            "dp": [0.0],
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

    for (
        activelearning_kwargs,
        datamodule_kwargs,
        model_kwargs,
        trainer_kwargs,
    ) in product(
        activelearning_kwargs_list,
        datamodule_kwargs_list,
        model_kwargs_list,
        trainer_kwargs_list,
    ):
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)

        text_selector_cls = activelearning_kwargs["text_selector_cls"]
        text_selector = text_selector_cls(
            class_dims=data_module.classes_dimensions,
            annotation_columns=data_module.annotation_columns,
            amount_per_user=activelearning_kwargs["amount_per_user"],
        )
        activelearning_kwargs["text_selector"] = text_selector

        trainer_kwargs["custom_callbacks"] = [
            PersonalizedMetricsCallback(),
        ]
        # TODO: Parametrize, for now we don't have alternative
        model = Baseline(
            output_dim=sum(data_module.classes_dimensions),
            embedding_dim=50,
        )
        module = StandardActiveLearningFlow(
            dataset=data_module,
            datamodule_kwargs=datamodule_kwargs,
            model=model,
            train_kwargs=trainer_kwargs,
            wandb_project_name=wandb_project_name,
            **activelearning_kwargs,
        )

        module.experiment(**activelearning_kwargs)
