import json
from pathlib import Path
from typing import Any, Optional

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from torch import Tensor


class LogTrainingDynamics(Callback):
    def __init__(
        self,
        save_dir: Path,
        log_epoch_freq: int = 1,
    ) -> None:
        self.log_epoch_freq = log_epoch_freq

        self.train_logits = []
        self.train_labels = []
        self.train_guids = []

        self.save_dir = save_dir.joinpath('training_dynamics')
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.ready = True

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self.train_logits.append(outputs['logits'])
        self.train_labels.append(outputs['labels'])
        self.train_guids.append(outputs['guids'])

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logits = torch.cat(self.train_logits).float().tolist()
        labels = torch.cat(self.train_labels).int().tolist()
        guids = torch.cat(self.train_guids).int().tolist()

        with self.save_dir.joinpath(f'dynamics_epoch_{trainer.current_epoch}.jsonl').open(
            mode='w'
        ) as f:
            for guid, sample_logits, gold in zip(guids, logits, labels):
                sample = {
                    'guid': int(guid),
                    f'logits_epoch_{trainer.current_epoch}': sample_logits,
                    'gold': gold,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        self.train_logits.clear()
        self.train_labels.clear()
        self.train_guids.clear()
