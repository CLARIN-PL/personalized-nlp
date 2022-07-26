import os
from itertools import product
from pytorch_lightning.callbacks import EarlyStopping

from personalized_nlp.datasets.wiki.toxicity import ToxicityDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "WikiToxicityCartographyBalanced"
    datamodule_cls = ToxicityDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embeddings_type": ["xlmr"],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["420balanced"],
            "fold_nums": [10],
            "batch_size": [3000],
            "test_fold": list(range(10)),
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
            "model_type": ["baseline", "peb", "onehot"],
        }
    )

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
                "train_size": len(data_module.train_dataloader().dataset),
                "val_size": len(data_module.val_dataloader().dataset),
                "test_size": len(data_module.test_dataloader().dataset) 
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
                    callbacks.CartographySaveCallback(
                        dir_name=f'cartography_wiki_toxicity_model={trainer_kwargs["model_type"]}',
                        fold_num=datamodule_kwargs["test_fold"],
                        fold_nums=datamodule_kwargs["fold_nums"],
                    ),
                    EarlyStopping(monitor="valid_loss", mode="min", patience=3)
                ],
            )

            logger.experiment.finish()
