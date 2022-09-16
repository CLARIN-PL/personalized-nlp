from typing import *
from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class PredictionsCallback(Callback):
    
    def __init__(self, metric, *args, **kwargs) -> None:
        super(PredictionsCallback, self).__init__(*args, **kwargs)
        self.outputs = []
        self.df = None
        self.metric = metric
    
    def on_test_batch_end(self, 
                           trainer: pl.Trainer, 
                           pl_module: pl.LightningModule, 
                           outputs: Dict[Any, Any], 
                           batch: int, 
                           batch_idx: int, 
                           unused: int = 0
        ) -> None:
        self.outputs.append(outputs)
    
    def on_test_epoch_end(self, *args, **kwargs) -> None:
        dfs = []
        for suboutput in self.outputs:
            for text_id, user_id, logits, y_true in zip(
                suboutput['x']['text_ids'],
                suboutput['x']['annotator_ids'],
                suboutput['output'],
                suboutput['y'],
            ): 
                guid = f'{text_id}_{user_id}'
                metric = logits.detach().item()
            df = pd.DataFrame({'guid': [guid], self.metric: [metric]})
            dfs.append(df)
        self.df = pd.concat(dfs)
