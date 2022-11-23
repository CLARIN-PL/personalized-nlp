# TODO: Refactor!!!
from datetime import datetime

import pytorch_lightning as pl
import torch

from personalized_active_learning.datasets import BaseDataset
from personalized_nlp.learning.classifier import Classifier
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from settings import CHECKPOINTS_DIR

def train_test(
    datamodule: BaseDataset,
    model,
    epochs=6,
    lr=1e-2,
    use_cuda=False,
    logger=None,
    custom_callbacks=None,
    monitor_metric="valid_loss",
    monitor_mode="min",
    advanced_output=False,
):
    """Train model and return predictions for test dataset"""
    text_embedding_dim = datamodule.text_embedding_dim

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    class_dims = datamodule.classes_dimensions
    class_names = datamodule.annotation_columns

    model = Classifier(
        model=model, lr=lr, class_dims=class_dims, class_names=class_names
    )

    if logger is not None:
        checkpoint_dir = CHECKPOINTS_DIR / logger.experiment.name
    else:
        datetime_now_string = datetime.now().strftime("%H:%M:%S")
        checkpoint_dir = CHECKPOINTS_DIR / f"checkpoint-{datetime_now_string}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, save_top_k=1, monitor=monitor_metric, mode=monitor_mode
    )
    progressbar_checkpoint = TQDMProgressBar(refresh_rate=20)

    _use_cuda = use_cuda and torch.cuda.is_available()

    callbacks = [checkpoint_callback, progressbar_checkpoint]
    if custom_callbacks is not None:
        callbacks = callbacks + custom_callbacks

    trainer = pl.Trainer(
        gpus=1 if _use_cuda else 0,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=2,
    )
    trainer.fit(model, train_loader, val_loader)
    train_metrics = trainer.logged_metrics
    trainer.test(dataloaders=test_loader)

    if advanced_output:
        return {"trainer": trainer, "train_metrics": train_metrics}
    else:
        return trainer