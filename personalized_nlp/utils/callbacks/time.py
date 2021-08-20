from pytorch_lightning.callbacks import Callback
import time


class TimingCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.epoch_stats = []

    def on_validation_epoch_end(self, trainer, pl_module, unused=None):
        current_train_time = time.time() - self.start_time
        epoch_number = trainer.current_epoch
        loss_value = trainer.callback_metrics['valid_loss'].cpu().item()

        self.epoch_stats.append((epoch_number, current_train_time, loss_value))