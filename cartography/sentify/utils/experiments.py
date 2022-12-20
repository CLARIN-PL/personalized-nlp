from pathlib import Path
from typing import Any, Optional

import torch
import wandb
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
)

from sentify.callbacks.time import (
    EpochTrainDurationCallback,
    TrainDurationCallback,
    TestDurationCallback,
)
from sentify.callbacks.training_dynamics import LogTrainingDynamics
from sentify.callbacks.wandb import WatchModel, LogParamsFile, LogConfusionMatrix
from sentify.data.augmentations.user_identifier import AUGMENTATIONS
from sentify.data.augmentations.user_retriever import (
    BiencoderComputeSimilarity,
    CrossEncoderComputeSimilarity,
)
from sentify.datasets import DATASETS
from sentify.datasets.base_datamodule import BaseDataModule
from sentify.models.models import TransformerSentimentModel, RetrieverModel
from sentify.models.retriever import Retriever
from sentify.utils.downloading import safe_from_pretrained


def run_experiment(
    config: dict[str, Any],
    model_trainer: LightningModule,
    datamodule: BaseDataModule,
    wandb_logger: WandbLogger,
    exp_dir: Path,
):
    checkpoint_dir = exp_dir.joinpath('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[
            WatchModel(log_graph=False),
            LogParamsFile(),
            LogConfusionMatrix(
                num_classes=datamodule.num_classes,
                log_modes=('test',),
            ),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                **config['checkpoint'],
            ),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(**config['early_stopping']),
            EpochTrainDurationCallback(),
            TrainDurationCallback(),
            TestDurationCallback(),
            LogTrainingDynamics(save_dir=exp_dir),
        ],
        log_every_n_steps=5,
        **config['trainer'],
    )

    wandb.require(experiment="service")
    trainer.fit(
        model=model_trainer,
        datamodule=datamodule,
    )

    metrics, *_ = trainer.test(dataloaders=datamodule)
    wandb_logger.log_metrics({k: v for k, v in metrics.items()})

    wandb_logger.experiment.finish()


def create_datamodule(config):
    tokenizer = safe_from_pretrained(AutoTokenizer, config['model']['name'])

    config = config['datamodule']
    dataset_name = config['dataset']
    datamodule_class = DATASETS[dataset_name]
    datamodule = datamodule_class(
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['seed'],
        max_length=config['max_length'],
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def create_user_identifier_datamodule(config):
    datamodule = create_datamodule(config)
    tokenizer = safe_from_pretrained(AutoTokenizer, config['model']['name'])
    ui_config = config['user_identifier']
    augmentation = AUGMENTATIONS[ui_config['augmentations']](tokenizer, ui_config['prefix_length'])
    datamodule.augmentations = augmentation
    return datamodule


def create_retriever_datamodule(config, logger: Optional[WandbLogger] = None):
    config_gpu = config['trainer']['gpus']
    device = f"cuda:{config_gpu[0]}" if config_gpu and torch.cuda.is_available() else 'cpu'

    datamodule = create_datamodule(config)

    retriever_config = config['retriever']
    encoder_name = retriever_config['encoder_name']
    if encoder_name.startswith('sentence-transformers'):
        augmentation = BiencoderComputeSimilarity(
            datamodule=datamodule,
            encoder=SentenceTransformer(
                model_name_or_path=encoder_name,
                device=device,
            ),
            name=encoder_name,
            logger=logger,
        )
    elif encoder_name.startswith('cross-encoder'):
        augmentation = CrossEncoderComputeSimilarity(
            datamodule=datamodule,
            encoder=CrossEncoder(
                model_name=encoder_name,
                device=device,
            ),
            name=encoder_name,
            logger=logger,
        )
    else:
        raise ValueError(encoder_name)

    datamodule.augmentations = augmentation
    return datamodule


def create_baseline_model(config, num_labels):
    config = config['model']
    seed_everything(config['seed'])

    model = safe_from_pretrained(
        AutoModelForSequenceClassification,
        config['name'],
        num_labels=num_labels,
    )
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    if config['freeze_backbone']:
        _freeze(model)

    return TransformerSentimentModel(
        model=model,
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        num_classes=num_labels,
    )


def create_retriever_model(config, num_labels):
    base_config = config['model']
    retriever_config = config['retriever']

    seed_everything(base_config['seed'])

    backbone = safe_from_pretrained(
        AutoModel,
        base_config['name'],
        add_pooling_layer=False,
    )
    model = Retriever(
        num_labels=num_labels,
        backbone=backbone,
        normalize_weights=retriever_config['normalize_weights'],
        feature_normalization=retriever_config['feature_normalization'],
        top_k=retriever_config['top_k'],
    )

    if base_config['gradient_checkpointing']:
        model.backbone.gradient_checkpointing_enable()

    if base_config['freeze_backbone']:
        _freeze(model.backbone)

    return RetrieverModel(
        model=model,
        learning_rate=base_config['learning_rate'],
        warmup_steps=base_config['warmup_steps'],
        num_classes=num_labels,
    )


def _freeze(model):
    model.base_model.apply(lambda param: param.requires_grad_(False))
