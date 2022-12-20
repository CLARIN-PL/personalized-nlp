import time
from datetime import timedelta

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback


class EpochTrainDurationCallback(Callback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_time = time.monotonic()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        end_time = time.monotonic()
        diff = timedelta(seconds=end_time - self.start_time)
        diff_seconds = diff.total_seconds()

        trainer.logger.log_metrics({f'train/time_epoch_{trainer.current_epoch}': diff_seconds})


class TrainDurationCallback(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        self.start_time = time.monotonic()

    def on_train_end(self, trainer, pl_module, unused=None):
        end_time = time.monotonic()
        diff = timedelta(seconds=end_time - self.start_time)
        diff_seconds = diff.total_seconds()
        trainer.logger.log_metrics({'train/time': diff_seconds})


class TestDurationCallback(Callback):
    def on_test_start(self, trainer, pl_module) -> None:
        self.start_time = time.monotonic()

    def on_test_end(self, trainer, pl_module, unused=None):
        end_time = time.monotonic()
        diff = timedelta(seconds=end_time - self.start_time)
        diff_seconds = diff.total_seconds()
        trainer.logger.log_metrics({'test/time': diff_seconds})
