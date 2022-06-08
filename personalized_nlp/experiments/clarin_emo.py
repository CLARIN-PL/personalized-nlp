import os
from itertools import product
from pytorch_lightning.callbacks import EarlyStopping

from personalized_nlp.datasets.clarin_emo_sent import ClarinEmoSentNoNoiseDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "Kartografia"
    datamodule_cls = ClarinEmoSentNoNoiseDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embedding_types": ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][
                :1
            ],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users", "texts"][1:],
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
            # "model_type": ["baseline", "onehot", "peb", "bias", "embedding"],
            "model_type": ["baseline"],
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
                    callbacks.CartographySaveCallback(
                        dir_name='clarin_emo_cartography'
                    ),
                    EarlyStopping(monitor="valid_loss", mode="min", patience=3),
                ],
            )

            logger.experiment.finish()
