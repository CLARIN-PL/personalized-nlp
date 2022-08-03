import pytorch_lightning as pl
import torch
from personalized_nlp.learning.flow_model import FlowModel
from personalized_nlp.models.flow_models import FLOW_MODELS_DICT
from settings import CHECKPOINTS_DIR, LOGS_DIR
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


def flow_train_test(
    datamodule,
    flow_type,
    flow_kwargs,
    model_type,
    model_kwargs,
    epochs=6,
    lr=1e-4,
    use_cuda=True,
    logger=None,
    log_model=False,
    custom_callbacks=None,
    monitor_metric="valid_loss",
    monitor_mode="min",
    **kwargs
):
    """Train model and return predictions for test dataset"""
    output_dim = len(datamodule.class_dims)
    text_embedding_dim = datamodule.text_embedding_dim

    flow_kwargs['features'] = output_dim
    flow_kwargs['context_features'] = text_embedding_dim 
    
    flow_model = FLOW_MODELS_DICT[model_type](
        annotator_num=datamodule.annotators_number,
        flow_type=flow_type,
        flow_kwargs=flow_kwargs,
        bias_vector_length=len(datamodule.class_dims),
        **model_kwargs
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    #raise Exception(f'Train: {len(train_loader.dataset)}\nVal: {len(val_loader.dataset)}\nTest: {len(test_loader.dataset)}\nSum: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}\nInner: {len(datamodule.annotations)}')
    
    class_names = datamodule.annotation_columns

    model = FlowModel(
        flow_model=flow_model, 
        lr=lr, 
        class_names=class_names
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
