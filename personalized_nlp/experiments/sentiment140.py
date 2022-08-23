import os
from itertools import product

from personalized_nlp.datasets.sentiment140 import Sentiment140DataModule
from personalized_nlp.utils.experiments import product_kwargs

from settings import LOGS_DIR
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from personalized_nlp.learning.train import train_test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "2022_wein_icdmw"
    datamodule_cls = Sentiment140DataModule

    datamodule_kwargs_list = product_kwargs({
        "regression": [False],
        "embeddings_type": ["roberta"],
        "stratify_folds_by": ["predefined"],
        "folds_num": [1],
        "batch_size": [32],
        "seed": list(range(42, 52)),
        "use_cuda": [True]
    })
    model_kwargs_list = product_kwargs({
        "embedding_dim": [50],
        "dp_emb": [0.25],
        "dp": [0.0],
        "hidden_dim": [100],
        "append_annotator_ids": [True]
    })
    trainer_kwargs_list = product_kwargs({
        "epochs": [50],
        "lr_rate": [0.008],
        "regression": [False],
        "use_cuda": [True],  # False
        "model_type": ["baseline", "onehot", "peb", "bias", "embedding"],
        "monitor_metric": ["valid_macro_f1_sentiment"],
        "monitor_mode": ["max"],
    })

    for (
            datamodule_kwargs,
            model_kwargs,
            trainer_kwargs
    ) in product(
            datamodule_kwargs_list,
            model_kwargs_list,
            trainer_kwargs_list
    ):
        data_module = datamodule_cls(**datamodule_kwargs)

        hparams = {
            "dataset": type(data_module).__name__,
            **datamodule_kwargs,
            **model_kwargs,
            **trainer_kwargs
        }

        logger = pl_loggers.WandbLogger(
            save_dir=str(LOGS_DIR),
            config=hparams,
            project=wandb_project_name,
            log_model=False,
        )

        train_test(
            datamodule=data_module,
            model_kwargs=model_kwargs,
            logger=logger,
            **trainer_kwargs,
            custom_callbacks=[
                EarlyStopping(monitor="valid_macro_f1_sentiment", mode="max", patience=5),
            ],
        )

        logger.experiment.finish()
