import os
from itertools import product

from active_learning.module import ActiveLearningModule
import active_learning.algorithms as algorithms
from personalized_nlp.datasets.wiki.aggression import AggressionDataModule

from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from personalized_nlp.utils.callbacks.personal_metrics import (
    PersonalizedMetricsCallback,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "Aggression_amount_per_user"
    datamodule_cls = AggressionDataModule

    activelearning_kwargs_list = product_kwargs(
        {
            "text_selector_cls": [
                # algorithms.TextAnnotationDiversitySelector,
                algorithms.RandomSelector,
                algorithms.ConfidenceSelector,
                algorithms.MaxPositiveClassSelector,
                algorithms.ConfidenceAllDimsSelector,
                algorithms.Confidencev2Selector,
                # algorithms.RandomImprovedSelector,
            ],
            "max_amount": [50_000],
            "step_size": [2_000],
            "amount_per_user": [None],
            "stratify_by_user": [False],
        }
    )
    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embeddings_type": ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][
                :1
            ],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users", "texts"][1:],
            "folds_num": [5],
            "batch_size": [3000],
            "test_fold": list(range(5)),
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "min_annotations_per_user_in_fold": [20],
        }
    )
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
            "regression": [False],
            "use_cuda": [False],
            "model_type": ["peb"],
            "monitor_metric": ["valid_macro_f1_mean"],
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
            class_dims=data_module.class_dims,
            annotation_columns=data_module.annotation_columns,
            amount_per_user=activelearning_kwargs["amount_per_user"],
        )
        activelearning_kwargs["text_selector"] = text_selector

        trainer_kwargs["custom_callbacks"] = [
            PersonalizedMetricsCallback(),
        ]

        module = ActiveLearningModule(
            datamodule=data_module,
            datamodule_kwargs=datamodule_kwargs,
            model_kwargs=model_kwargs,
            train_kwargs=trainer_kwargs,
            wandb_project_name=wandb_project_name,
            **activelearning_kwargs,
        )

        module.experiment(**activelearning_kwargs)
