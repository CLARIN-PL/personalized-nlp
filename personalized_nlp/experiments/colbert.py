import os
from itertools import product

from personalized_nlp.datasets.colbert import ColbertDataModule
from personalized_nlp.utils.experiments import product_kwargs

from settings import LOGS_DIR
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from personalized_nlp.learning.train import train_test
from personalized_nlp.utils.callbacks import (SaveOutputsLocal,
                                              LogTrainingDynamics,
                                              PersonalizedMetricsCallback)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "chatgpt_colbert"
    datamodule_cls = ColbertDataModule

    datamodule_kwargs_list = product_kwargs({
        "regression": [False],
        "embeddings_type": ["roberta"],
        "stratify_folds_by": ["predefined"],
        "folds_num": [1],
        "batch_size": [32],  # [10] for UserId model, [32] otherwise
        "seed": list(range(42, 52))[:1],
        "use_cuda": [True],
        "use_finetuned_embeddings":
        [False]  # [False] for UserId model, [True] otherwise
    })
    model_kwargs_list = product_kwargs({
        "embedding_dim": [50],
        "dp_emb": [0.25],
        "dp": [0.0],
        "hidden_dim": [100],
        "append_annotator_ids":
        [True]  # [True] for UserId model, [False] otherwise
    })
    trainer_kwargs_list = product_kwargs({
        "epochs":
        [50
         ],  # [3] for UserId model or [10] with early stopping, [50] otherwise
        # "lr_rate": [0.00001],  # [0.00001] for UserId model, [0.008] otherwise
        "lr": [0.008],  # [0.00001] for UserId model, [0.008] otherwise
        # "auto_lr_find": [True],
        "regression": [False],
        "use_cuda": [True],  # False
        "model_type": [
            "baseline", "onehot", "peb", "bias", "embedding",
            "transformer_user_id"
        ][:-1],
        "monitor_metric": [f'valid_macro_f1_{"is_funny"}'],
        "monitor_mode": ["max"],
    })

    for (datamodule_kwargs, model_kwargs,
         trainer_kwargs) in product(datamodule_kwargs_list, model_kwargs_list,
                                    trainer_kwargs_list):
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
                EarlyStopping(
                    monitor=
                    f"valid_macro_f1_{data_module.annotation_columns[0]}",
                    mode="max",
                    patience=5),
                SaveOutputsLocal(save_dir=str(LOGS_DIR),
                                 model=trainer_kwargs["model_type"],
                                 dataset=type(data_module).__name__,
                                 seed=datamodule_kwargs["seed"]),
                # LogTrainingDynamics(save_dir=LOGS_DIR / 'training_dynamics' /
                #                     str(type(data_module).__name__) /
                #                     trainer_kwargs["model_type"] /
                #                     str(data_module.annotation_columns[0])),
                PersonalizedMetricsCallback()
            ],
        )

        logger.experiment.finish()
