from pytorch_lightning.callbacks import ModelCheckpoint


class FreezingCallback(ModelCheckpoint):
    def __init__(self, freeze_epochs_ratio: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.freeze_epochs_ratio = freeze_epochs_ratio

    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not hasattr(pl_module.model, "set_requires_grad"):
            return super().on_epoch_end(trainer, pl_module)

        if trainer.current_epoch % 2 == 0:
            pl_module.model.set_requires_grad("text", True)
            pl_module.model.set_requires_grad("user", False)
            trainer.datamodule.train_with_major_votes = True
            trainer.reset_train_dataloader()

        return super().on_epoch_start(trainer, pl_module)

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if not hasattr(pl_module.model, "set_requires_grad"):
            return super().on_epoch_end(trainer, pl_module)

        max_epochs = trainer.max_epochs
        current_epoch = trainer.current_epoch
        if current_epoch % 2 == 1:
            pl_module.model.set_requires_grad("text", False)
            pl_module.model.set_requires_grad("user", True)
            trainer.datamodule.train_with_major_votes = False
            trainer.reset_train_dataloader()

        return super().on_epoch_end(trainer, pl_module)
