import os
import functools
from itertools import product
from pytorch_lightning.callbacks import EarlyStopping

from personalized_nlp.datasets.wiki.aggression import AggressionDataModule

from personalized_nlp.learning.train import train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
import personalized_nlp.utils.callbacks as callbacks
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers
from personalized_nlp.utils import prune_train_set, create_folds_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

if __name__ == "__main__":
    wandb_project_name = "WikiAmbigous"
    datamodule_cls = AggressionDataModule

    datamodule_kwargs_list = product_kwargs(
        {
            "regression": [False],
            "embeddings_type": ["labse", "mpnet", "xlmr", "random", "skipgram", "cbow"][
                :1
            ],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["texts"],
            "fold_nums": [10],
            "batch_size": [3000],
            "test_fold": list(range(10)),
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
            "filter_annotations_path": [
                "/home/konradkaranowski/storage/personalized-nlp/storage/outputs/cartography_outputs/cartography_wiki_agr_model=onehot/filtered/class_aggression"
            ],
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
            "model_type": ["onehot"],
            "top_perc": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    )

    for datamodule_kwargs in datamodule_kwargs_list:
        seed_everything()
        prune_dict = create_folds_dict(root='/home/konradkaranowski/storage/personalized-nlp/storage/outputs/cartography_outputs/cartography_wiki_agr_model=onehot/metrics/class_aggression')
        data_module = datamodule_cls(**datamodule_kwargs)

        for model_kwargs, trainer_kwargs in product(
            model_kwargs_list,
            trainer_kwargs_list,
        ):
            prune_function = functools.partial(
                prune_train_set,
                metrics_dict=prune_dict,
                sort_by='variability', 
                ascending=True, 
                top_perc=trainer_kwargs['top_perc']
            )
            
            data_module.set_prune_train_function(prune_function)
            
            hparams = {
                "dataset": type(data_module).__name__,
                **datamodule_kwargs,
                **model_kwargs,
                **trainer_kwargs,
                "train_size": len(data_module.train_dataloader().dataset)
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
                **trainer_kwargs
            )

            logger.experiment.finish()
