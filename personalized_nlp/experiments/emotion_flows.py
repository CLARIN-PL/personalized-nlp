from ast import arg
import os
from itertools import product
from pytorch_lightning.callbacks import EarlyStopping

from personalized_nlp.datasets.emotions import EmotionsSimpleDataModule, EmotionsCollocationsDatamodule
from personalized_nlp.datasets.wiki import AggressionDataModule, ToxicityDataModule, AttackDataModule, AgressionAttackCombinedDatamodule
from personalized_nlp.datasets.clarin_emo_sent import ClarinEmoSentNoNoiseDataModule
from personalized_nlp.datasets.clarin_emo_text import ClainEmoTextNoNoiseDataModule

from personalized_nlp.utils.callbacks import SaveDistribution
from personalized_nlp.learning.train_flow import flow_train_test
from settings import LOGS_DIR
from personalized_nlp.utils import seed_everything
from personalized_nlp.utils.experiments import product_kwargs
from pytorch_lightning import loggers as pl_loggers
import argparse
from personalized_nlp.experiments.grid_file import GRIDS

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--grid',
        '-g',
        type=str,
        required=True,
        help='Name of grid key',
        dest='grid'
    )
    return parser.parse_args()


if __name__ == "__main__":
    wandb_project_name = "AggressionAttack2"
    used_folds = 10
    datamodule_classes = [
        AgressionAttackCombinedDatamodule
    ]
    
    #grid_name = parse_args().grid
    #grid = GRIDS[grid_name]

    datamodule_kwargs_list = product_kwargs(
        {
            "embeddings_type": ["labse"],
            "limit_past_annotations_list": [None],
            "stratify_folds_by": ["users"],
            "fold_nums": [10],
            "batch_size": [3000],
            "test_fold": list(range(used_folds)),
            "use_finetuned_embeddings": [False],
            "major_voting": [False],
        }
    )
    # model_kwargs_list = grid['model_kwargs_list']
    # flow_kwargs_list = grid['flow_kwargs_list']
    # trainer_kwargs_list = grid['trainer_kwargs_list']
    
    model_kwargs_list = product_kwargs(
        {
            "embedding_dim": [50],
            "dp_emb": [0.25],
            "dp": [0.0],
            "hidden_dim": [100],
        }
    )
    flow_kwargs_list = product_kwargs(
        {
            "hidden_features": [2], #[2, 4, 8],
            "num_layers":  [3], #[1, 2, 3],
            "num_blocks_per_layer":  [2], #[1, 2, 4],
            "dropout_probability": [0.3], #[0.0, 0.1, 0.3],
            "batch_norm_within_layers": [True], #[True, False],
            "batch_norm_between_layers":  [True], #[True, False],
        }
    )
    trainer_kwargs_list = product_kwargs(
        {
            "epochs": [500],
           "lr_rate": [1e-4],
            "use_cuda": [True],
            "flow_type": [
               # 'maf', 
                'real_nvp', 
                #'nice'
            ],#["nice", "maf", "real_nvp"],
            "model_type": [
                'flow_baseline', 
                'flow_onehot',
                'flow_peb',
                'flow_bias'
            ]
        }
    )

    for datamodule_cls in datamodule_classes:
        for datamodule_kwargs in datamodule_kwargs_list:
            seed_everything()
            data_module = datamodule_cls(**datamodule_kwargs)

            for model_kwargs, trainer_kwargs, flow_kwargs in product(
                model_kwargs_list,
                trainer_kwargs_list,
                flow_kwargs_list
            ):
                hparams = {
                    "dataset": type(data_module).__name__,
                    #"grid_name": grid_name,
                    **datamodule_kwargs,
                    **model_kwargs,
                    **trainer_kwargs,
                    **flow_kwargs
                }

                logger = pl_loggers.WandbLogger(
                    save_dir=LOGS_DIR,
                    config=hparams,
                    project=wandb_project_name,
                    log_model=False,
                )

                flow_train_test(
                    datamodule=data_module,
                    model_kwargs=model_kwargs,
                    flow_kwargs=flow_kwargs,
                    logger=logger,
                    **trainer_kwargs,
                    custom_callbacks=[
                        EarlyStopping(monitor="valid_loss", mode="min", patience=10),
                        #SaveDistribution(save_dir='distro', fold=datamodule_kwargs["test_fold"])
                    ],
                )

                logger.experiment.finish()
