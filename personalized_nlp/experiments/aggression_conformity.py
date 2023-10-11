import os
from itertools import product
from pytorch_lightning.callbacks import EarlyStopping

from personalized_nlp.datasets.wiki.aggression import AggressionDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "AggressionConformity"
    datamodule_cls = AggressionDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embeddings_type": ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][
                :1
            ],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users", "texts"][1:],
            "fold_nums": [5],
            "batch_size": [3000],
            "test_fold": list(range(5)),
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
            "features_vector_length": [26],
        }
    )
    trainer_kwargs_list = product_kwargs(
        {
            "epochs": [20],
            "lr": [0.008],
            "regression": [False],
            "use_cuda": [True],
            # "model_type": ["baseline", "onehot", "peb", "bias", "embedding"],
            "model_type": ["conformity", "baseline", "peb", "bias"],
        }
    )

    for datamodule_kwargs in datamodule_kwargs_list:
        seed_everything()
        data_module = datamodule_cls(**datamodule_kwargs)

        for model_kwargs, trainer_kwargs in product(
            model_kwargs_list,
            trainer_kwargs_list,
        ):
            model_kwargs[
                "features_vector_length"
            ] = data_module.annotator_conformities.values.shape[1]
            hparams = {
                "dataset": type(data_module).__name__,
                **datamodule_kwargs,
                **model_kwargs,
                **trainer_kwargs,
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
            )

            logger.experiment.finish()
