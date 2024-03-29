import os
from itertools import product
from datetime import datetime

from personalized_nlp.datasets.humicroedit.humicroedit_majority import (
    HumicroeditMajorityDataModule)
from personalized_nlp.utils.callbacks import (SaveOutputsLocal,
                                              PersonalizedMetricsCallback)

from settings import LOGS_DIR
from personalized_nlp.learning.train import train_test
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_DIR"] = str(LOGS_DIR)

if __name__ == "__main__":
    wandb_project_name = "MergedHumorDatasets"
    datamodule_cls = HumicroeditMajorityDataModule

    datamodule_kwargs_list = product_kwargs({
        "use_cuda": [True],
        "regression": [False],
        "embeddings_type":
        ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][:1],
        "limit_past_annotations_list": [None],
        "stratify_folds_by": ["users", "texts"][1:],
        "folds_num": [10],
        # batch_size = [10] for UserId model, [32] otherwise, [3000] as default
        "batch_size": [10],
        "use_finetuned_embeddings":
        [False],  # [False] for UserId model, [True] otherwise
        "seed":
        list(range(42, 52))[:1],
        "test_fold":
        list(range(10)),
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
        # [3] for UserId model or [10] for UserId model with early stopping, [20] otherwise
        "epochs": [3],
        # [0.00001] or [8e-6] for UserId model, [0.008] otherwise
        "lr": [0.00001],
        "regression": [False],
        "use_cuda": [True],
        "model_type": ["baseline", "onehot", "peb", "bias", "embedding",
                       "transformer_user_id"][-1:],
    })

    for datamodule_kwargs in datamodule_kwargs_list:
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)

        for model_kwargs, trainer_kwargs in product(
                model_kwargs_list,
                trainer_kwargs_list,
        ):
            hparams = {
                "dataset": type(data_module).__name__,
                **datamodule_kwargs,
                **model_kwargs,
                **trainer_kwargs,
            }

            logger = pl_loggers.WandbLogger(
                save_dir=LOGS_DIR,
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
                    PersonalizedMetricsCallback(),
                    EarlyStopping(
                        monitor="valid_personal_macro_f1",
                        mode="max",
                        patience=5),
                    SaveOutputsLocal(save_dir=(str(LOGS_DIR /
                                                   f"{datetime.now().strftime('%m-%d-%Y-%h')}_{str(type(data_module).__name__)}" /
                                                   trainer_kwargs["model_type"])),
                                     model=trainer_kwargs["model_type"],
                                     dataset=type(data_module).__name__,
                                     seed=datamodule_kwargs["seed"],
                                     test_fold=datamodule_kwargs["test_fold"])
                ])

            logger.experiment.finish()
