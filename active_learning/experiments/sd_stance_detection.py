import os
from itertools import product

from active_learning.module import ActiveLearningModule
import active_learning.algorithms as algorithms
from personalized_nlp.datasets.stance_detection.sd import (
    SDStanceDetectionDataModule,
)

from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from personalized_nlp.utils.callbacks.personal_metrics import (
    PersonalizedMetricsCallback,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "20"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "SD_stance_detection_AL_reduced_dims"
    datamodule_cls = SDStanceDetectionDataModule

    activelearning_kwargs_list = product_kwargs(
        {
            "text_selector_cls": [
                algorithms.RandomSelector,
                algorithms.ConfidenceSelector,
                # algorithms.Confidencev2Selector,
                # algorithms.ConfidenceAllDimsSelector,
                # algorithms.MaxPositiveClassSelector,
                algorithms.BalancedClassesPerTextSelector,
                algorithms.BalancedClassesPerUserSelector,
                algorithms.BalancedConfidenceSelector,
            ],
            "max_amount": [10_000],
            "step_size": [480],
            "amount_per_user": [100],
            "stratify_by_user": [True],
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
            "batch_size": [500],
            "test_fold": list(range(5)),
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "min_annotations_per_user_in_fold": [None],
            "test_batch_size": [1000],
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
            "lr_rate": [0.008],
            "regression": [False],
            "use_cuda": [True],
            "model_type": ["peb"],
            # "model_type": ["baseline", "peb"],
            "monitor_metric": ["valid_loss"],
            # "monitor_metric": ["valid_macro_f1_mean"],
            "monitor_mode": ["min"],
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
