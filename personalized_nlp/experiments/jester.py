import os
from itertools import product

from personalized_nlp.datasets.jester.jester import JesterDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "Jester_new"
    datamodule_cls = JesterDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embeddings_type": ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users"],
            "fold_nums": [10],
            "batch_size": [3000],
            "test_fold": list(range(10))[:1],
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
            "model_type": ["baseline", "onehot", "peb", "bias", "embedding"],
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
                **trainer_kwargs
            )

            logger.experiment.finish()
