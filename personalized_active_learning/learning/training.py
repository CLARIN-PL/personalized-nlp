from datetime import datetime
from typing import Optional, List
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, Callback

from personalized_nlp.learning.classifier import Classifier
from personalized_active_learning.models import IModel
from settings import CHECKPOINTS_DIR


def train_test(
    model: IModel,
    datamodule,  # type is BaseDataset, cannot be imported due to circular import-finetune
    logger: WandbLogger,
    epochs: int = 6,
    lr: float = 1e-2,
    use_cuda: bool = False,
    monitor_metric: str = "valid_loss",
    monitor_mode: str = "min",
    custom_callbacks: Optional[List[Callback]] = None
) -> Trainer:
    """Train model and return predictions for test dataset

    A new model is created as a `Classifier` from `personalized_nlp.learning.classifier`
    with layers of `model`. Then `Trainer` is created and data is fitted and tested.

    Args:
        model (IModel):  A model defined in `personalized_active_learning.models`
                directory. Used for a main training. (Imorted definition, eg. `Baseline`).
        datamodule (_type_): A dataset which derives from the
                `personalized_active_learning.datasets.base` class.
        logger (WandbLogger): Instantion of prepared `WandbLogger`.
        epochs (int, optional): Number of training epochs. Defaults to 6.
        lr (float, optional): Learning rate for a model. Defaults to 1e-2.
        use_cuda (bool, optional): Use cuda. Extra check is done with
            `torch.cuda.is_available()`. Defaults to False.
        monitor_metric (str, optional): Monitor argument for `ModelCheckpoint`.
                https://keras.io/api/callbacks/model_checkpoint/. Defaults to "valid_loss"
        monitor_mode (str, optional): Mode argument for `ModelCheckpoint`
                  https://keras.io/api/callbacks/model_checkpoint/. Defaults to "min".
        custom_callbacks (Optional[List[Callback]], optional): A list with custom
            callbacks for `Trainer`. Defaults to None.

    Returns:
        Trainer: Trained model in `Trainer` instance.
        https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    """

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    class_dims = datamodule.classes_dimensions
    class_names = datamodule.annotation_columns

    model = Classifier(model=model, lr=lr, class_dims=class_dims, class_names=class_names)

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

    trainer = Trainer(
        gpus=1 if _use_cuda else 0,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=2,
    )
    trainer.fit(model, train_loader, val_loader)
    _ = trainer.logged_metrics  # we can extract losses here
    trainer.test(dataloaders=test_loader, ckpt_path="best")

    return trainer
