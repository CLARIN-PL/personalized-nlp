from typing import *
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TrainDynamicsCallback(Callback):
    
    def __init__(self, *args, **kwargs) -> None:
        super(TrainDynamicsCallback, self).__init__(*args, **kwargs)
        self.outputs = []
        self.epoch = 0
        self.training_dynamics = defaultdict(
            lambda: defaultdict(list)
        )
    
    def on_train_batch_end(self, 
                           trainer: pl.Trainer, 
                           pl_module: pl.LightningModule, 
                           outputs: Dict[Any, Any], 
                           batch: int, 
                           batch_idx: int, 
                           unused: int = 0
        ) -> None:
        self.outputs.append(outputs)
    
    def on_train_epoch_end(self, *args, **kwargs) -> None:
        for suboutput in self.outputs:
            for text_id, user_id, logits, y_true in zip(
                suboutput['x']['text_ids'],
                suboutput['x']['annotator_ids'],
                suboutput['output'],
                suboutput['y'],
            ): 
                guid = f'{text_id}_{user_id}'
                gold = y_true.int()
                logits = logits.tolist()
                self.training_dynamics[guid]['gold'] = gold
                self.training_dynamics[guid]['logits'].append(logits)
        self.epoch += 1
        self.outputs = []
