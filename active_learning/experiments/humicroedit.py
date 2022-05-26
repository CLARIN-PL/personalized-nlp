import os
from itertools import product

from active_learning.module import ActiveLearningModule
from active_learning.algorithms.random import RandomSelector
from active_learning.algorithms.confidence import ConfidenceSelector
from personalized_nlp.datasets.humicroedit.humicroedit import HumicroeditDataModule

from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from personalized_nlp.utils.callbacks.personal_metrics import (
    PersonalizedMetricsCallback, )

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "Humicroedit_ActiveLearning_10fold"
    datamodule_cls = HumicroeditDataModule

    activelearning_kwargs_list = product_kwargs({
        "text_selector_cls": [RandomSelector, ConfidenceSelector],
        "max_amount": [100_000],
        "step_size": [5000],
    })
    datamodule_kwargs_list = product_kwargs({
        "regression": [False],
        "embedding_types":
        ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][:1],
        "limit_past_annotations_list": [None],
        "stratify_folds_by": ["users", "texts"][1:],
        "fold_nums": [10],
        "batch_size": [3000],
        "fold_num":
        list(range(10))[:5][:1],
        "use_finetuned_embeddings": [False],
        "major_voting": [False],
    })
    model_kwargs_list = product_kwargs({
        "embedding_dim": [50],
        "dp_emb": [0.25],
        "dp": [0.0],
        "hidden_dim": [100],
    })
    trainer_kwargs_list = product_kwargs({
        "epochs": [20],
        "lr_rate": [0.008],
        "regression": [False],
        "use_cuda": [False],
        "model_type": ["baseline", "onehot", "embedding"],
        "monitor_metric": ["valid_f1_is_funny_1"],
        "monitor_mode": ["max"],
    })

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
        text_selector = text_selector_cls(class_dims=data_module.class_dims)
        activelearning_kwargs["text_selector"] = text_selector

        trainer_kwargs["custom_callbacks"] = [PersonalizedMetricsCallback()]

        module = ActiveLearningModule(
            datamodule=data_module,
            datamodule_kwargs=datamodule_kwargs,
            model_kwargs=model_kwargs,
            train_kwargs=trainer_kwargs,
            wandb_project_name=wandb_project_name,
            **activelearning_kwargs,
        )

        module.experiment(**activelearning_kwargs)
