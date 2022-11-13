import copy

import pytorch_lightning as pl
import torch
from personalized_nlp.learning.classifier import Classifier
from personalized_nlp.learning.regressor import Regressor
from personalized_nlp.models import models as models_dict
from settings import CHECKPOINTS_DIR, LOGS_DIR
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from typing import Optional

def train_test(
    datamodule,
    model_kwargs,
    model_type,
    epochs=6,
    lr=1e-2,
    regression=False,
    use_cuda=False,
    logger=None,
    log_model=False,
    custom_callbacks=None,
    monitor_metric="valid_loss",
    monitor_mode="min",
    pretrained_model: Optional[torch.nn.Module] = None,
    **kwargs
) -> pl.Trainer:
    """Train model and return predictions for test dataset

    Args:
        datamodule: Source of training & test data.
        model_kwargs: Keyword arguments used to initialize model.
        model_type: Type of model as specified in models package.
        epochs: Max number of training epochs.
        lr: Learning rate applied during training.
        regression: Whether to use Regression or Classification.
        use_cuda: Whether CUDA should be used during training.
        logger: Pytorch Lighting logger used to log training process.
        log_model: Whether model should be logged.
        custom_callbacks: Custom callbacks that should be added to training callbacks.
        monitor_metric: Which metric is a primarily focus of monitoring.
        monitor_mode: Monitoring mode.
        pretrained_model: Pretrained model to use instead of `model_kwargs`.
        **kwargs:

    Returns:
        Pytorch Lighting `Trainer` instance.
    """
    output_dim = (
        len(datamodule.class_dims) if regression else sum(datamodule.class_dims)
    )
    text_embedding_dim = datamodule.text_embedding_dim
    # Use pretrained model if provided
    if pretrained_model:
        model = copy.deepcopy(pretrained_model)
    else:
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
    
    #raise Exception(f'Train: {len(train_loader.dataset)}\nVal: {len(val_loader.dataset)}\nTest: {len(test_loader.dataset)}\nSum: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}\nInner: {len(datamodule.annotations)}')

    if not pretrained_model:
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

    _use_cuda = use_cuda and torch.cuda.is_available()

    callbacks = [checkpoint_callback, progressbar_checkpoint]
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

    return trainer
