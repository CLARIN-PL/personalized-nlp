import os
from itertools import product

from active_learning.module import ActiveLearningModule
from active_learning.algorithms.random import random_selector
from personalized_nlp.datasets.wiki.aggression import AggressionDataModule

from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "Wiki_ActiveLearning"
    datamodule_cls = AggressionDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embedding_types": ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][
                :1
            ],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users", "texts"][:1],
            "fold_nums": [10],
            "batch_size": [3000],
            "fold_num": list(range(10))[:1],
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
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
            "use_cuda": [False],
            "model_type": ["baseline", "onehot", "bias"],
        }
    )

    for datamodule_kwargs, model_kwargs, trainer_kwargs in product(
        datamodule_kwargs_list, model_kwargs_list, trainer_kwargs_list
    ):
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)

        module = ActiveLearningModule(
            datamodule=data_module,
            text_selector=random_selector,
            datamodule_kwargs=datamodule_kwargs,
            model_kwargs=model_kwargs,
            train_kwargs=trainer_kwargs,
            wandb_project_name=wandb_project_name,
        )

        module.experiment(20_000, 1000)
