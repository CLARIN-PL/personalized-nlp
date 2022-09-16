from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from cartography_active_learning.learning import train_dynamics_callback
from personalized_nlp.learning.classifier import Classifier
from personalized_nlp.learning.regressor import Regressor
from personalized_nlp.models import models as models_dict
from settings import CHECKPOINTS_DIR, LOGS_DIR

from cartography_active_learning.datasets import CartographyDataModule
from cartography_active_learning.learning.train_dynamics_callback import TrainDynamicsCallback


def train_model_cartography(
    model_type: str,
    datamodule: CartographyDataModule,
    model_kwargs: Dict[str, Any],
    epochs=6,
    lr=1e-2,
    regression=False,
    use_cuda=True,
    logger=None,
    log_model=False,
    custom_callbacks=None,
    monitor_metric="valid_loss",
    monitor_mode="min",
    **kwargs
):
    """Train model and return predictions for test dataset"""
    output_dim = (
        len(datamodule.class_dims) if regression else sum(datamodule.class_dims)
    )
    text_embedding_dim = datamodule.text_embedding_dim
    model_cls = models_dict[model_type]

    model = model_cls(
        output_dim=output_dim,
        text_embedding_dim=text_embedding_dim,
        annotator_num=datamodule.annotators_number,
        bias_vector_length=len(datamodule.class_dims),
        **model_kwargs
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
        
    if regression:
        class_names = datamodule.annotation_columns

        model = Regressor(model=model, lr=lr, class_names=class_names)
    else:
        class_dims = datamodule.class_dims
        class_names = datamodule.annotation_columns

        model = Classifier(
            model=model, lr=lr, class_dims=class_dims, class_names=class_names
        )

    if logger is None:
        logger = pl_loggers.WandbLogger(save_dir=LOGS_DIR, log_model=log_model)

    checkpoint_dir = CHECKPOINTS_DIR / logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, save_top_k=1, monitor=monitor_metric, mode=monitor_mode
    )
    progressbar_checkpoint = TQDMProgressBar(refresh_rate=20)
    
    # cartography
    train_dynamics_callback = TrainDynamicsCallback()

    _use_cuda = use_cuda and torch.cuda.is_available()

    callbacks = [checkpoint_callback, progressbar_checkpoint, train_dynamics_callback]
    if custom_callbacks is not None:
        callbacks = callbacks + custom_callbacks

    trainer = pl.Trainer(
        gpus=1 if _use_cuda else 0,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)
    logger.experiment.finish()

    for callback in trainer.callbacks:
        if isinstance(callback, TrainDynamicsCallback):
            return callback.training_dynamics
    else:
        raise Exception('Wyjebało się XD')

