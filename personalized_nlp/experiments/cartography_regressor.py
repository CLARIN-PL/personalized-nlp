import os
import functools
from itertools import product
from pytorch_lightning.callbacks import EarlyStopping

from personalized_nlp.datasets.wiki.toxicity_cartography import ToxicityCartographyRegressorDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers
from personalized_nlp.utils.cartography_utils import prune_train_set, create_folds_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    sort_by = 'random'
    stratify_by_users = True
    wandb_project_name = "ToxicityRegressor10prcOutputs"
    datamodule_cls = ToxicityCartographyRegressorDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [True],
            "embeddings_type": ["xlmr"],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["420balanced"],
            "fold_nums": [3],
            "batch_size": [3000],
            "test_fold": [0],
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "model": ["baseline", "peb"],
            "value": ['variability']
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
            "epochs": [100],
            "lr_rate": [0.008],
            "regression": [True],
            "use_cuda": [True],
            "ascending": [True, False],
        }
    )

    for datamodule_kwargs in datamodule_kwargs_list:
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)

        for model_kwargs, trainer_kwargs in product(
            model_kwargs_list,
            trainer_kwargs_list,
        ):
            trainer_kwargs['model_type'] = datamodule_kwargs['model']
            hparams = {
                "dataset": type(data_module).__name__,
                **datamodule_kwargs,
                **model_kwargs,
                **trainer_kwargs,
                "ascending": True,
                "train_size": len(data_module.train_dataloader().dataset),
                "stratify_by_users": stratify_by_users
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
                    EarlyStopping(monitor="valid_loss", mode="min", patience=3),
                    callbacks.SaveOutputsLocal(save_dir='regressor_outputs', save_text=False, model=trainer_kwargs['model_type'], fold=datamodule_kwargs['test_fold'])
                ]
            )

            logger.experiment.finish()
