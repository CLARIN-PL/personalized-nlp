import datetime
import os

import torch.multiprocessing
from pytorch_lightning.loggers import WandbLogger

from sentify import EXPERIMENT_DIR
from sentify.utils.config import load_config
from sentify.utils.experiments import (
    run_experiment,
    create_baseline_model,
    create_datamodule,
    create_user_identifier_datamodule,
    create_retriever_datamodule,
    create_retriever_model,
)

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
METHODS = {
    'baseline': (create_baseline_model, create_datamodule),
    'user_identifier': (
        create_baseline_model,
        create_user_identifier_datamodule,
    ),
    'retriever': (create_retriever_model, create_retriever_datamodule),
}

config = load_config()
exp_method = config['method']
model_func, datamodule_func = METHODS[exp_method]

exp_data = config['datamodule']['dataset']
EXP_NAME = f'{exp_data}_{exp_method}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
exp_dir = EXPERIMENT_DIR.joinpath(EXP_NAME)

wandb_logger = WandbLogger(
    **config['wandb_logger'],
    name=EXP_NAME,
    save_dir=str(exp_dir),
)

wandb_logger.log_hyperparams(config)

if config['method'] == 'retriever':
    datamodule = datamodule_func(config, logger=wandb_logger)
else:
    datamodule = datamodule_func(config)

run_experiment(
    config,
    model_trainer=model_func(config, datamodule.num_classes),
    datamodule=datamodule,
    wandb_logger=wandb_logger,
    exp_dir=exp_dir,
)
