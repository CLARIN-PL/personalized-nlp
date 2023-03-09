from datetime import datetime

import pytorch_lightning as pl
import torch
from personalized_nlp.learning.classifier import Classifier
from personalized_nlp.learning.regressor import Regressor
from personalized_nlp.models import models as models_dict
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from settings import CHECKPOINTS_DIR, LOGS_DIR


def train_test(
    datamodule,
    model_kwargs,
    model_type,
    epochs=6,
    lr=1e-2,
    regression=False,
    use_cuda=False,
    logger=None,
    custom_callbacks=None,
    monitor_metric="valid_loss",
    monitor_mode="min",
    advanced_output=False,
    round_outputs=False,
    **kwargs,
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
        **model_kwargs,
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if regression:
        class_names = datamodule.annotation_columns

        model = Regressor(model=model, lr=lr, class_names=class_names, round_outputs=round_outputs)
    else:
        class_dims = datamodule.class_dims
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
    trainer.test(dataloaders=test_loader, ckpt_path="best")

    if advanced_output:
        return {"trainer": trainer, "train_metrics": train_metrics}
    else:
        return trainer
