import pytorch_lightning as pl

from transformers import get_linear_schedule_with_warmup


class TransformerLrScheduler(pl.Callback):
    def __init__(self, warmup_proportion: float = 0.0) -> None:
        super().__init__()
        self.warmup_proportion = warmup_proportion

    def on_train_start(self, trainer, pl_module):
        n_train = len(trainer.train_dataloader)
        n_accumulate_grad = trainer.accumulate_grad_batches
        n_max_epochs = trainer.max_epochs
        n_devices = trainer.num_gpus or 1

        num_training_steps = n_train // n_accumulate_grad * n_max_epochs // n_devices

        num_warmup_steps = int(self.warmup_proportion * num_training_steps)

        lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizers[0],
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        trainer.lr_schedulers = [{'scheduler': lr_scheduler, 'interval': "step", 'name': None, 'frequency': 1,
                                  'reduce_on_plateau': False, 'monitor': None, 'strict': True, 'opt_idx': None}]
