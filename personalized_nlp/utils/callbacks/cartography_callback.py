from typing import Any
import json
import os
from pytorch_lightning.callbacks import Callback

from personalized_nlp.settings import CARTOGRAPHY_OUTPUTS_DIR_NAME



class CartographySaveCallback(Callback):
    
    def __init__(
            self, 
            cartography_dir: str = 'cartography'
        ) -> None:
        super(CartographySaveCallback, self).__init__()
        self.outputs = []
        self.epoch = 0
        self.save_dir = CARTOGRAPHY_OUTPUTS_DIR_NAME / cartography_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    
    def get_guid(self, 
            text_id: int,
            user_id: int
        ) -> str:
        return f'text={text_id}_user={user_id}'

    
    def on_train_batch_end(self, 
                           trainer: "pl.Trainer", 
                           pl_module: "pl.LightningModule", 
                           outputs: dict, 
                           batch: int, 
                           batch_idx: int, 
                           unused: int = 0
        ) -> None:
        self.outputs.append(outputs)
        
    def on_train_epoch_end(self, *args, **kwargs):
        to_save = []
        for suboutput in self.outputs:
            for text_id, user_id, logits, y_true in zip(
                suboutput['x']['text_ids'],
                suboutput['x']['annotator_ids'],
                suboutput['output'],
                suboutput['y']
            ): 
                # raise Exception(f'Id: {text_id} user: {user_id} logits: {logits.shape}, y_true: {y_true.shape}')
                guid = self.get_guid(text_id, user_id)
                to_save.append({
                    'guid': guid,
                    f'logits_epoch_{self.epoch}': logits.tolist(),
                    'gold': y_true.item()
                })
        json_path = os.path.join(self.save_dir, f'dynamics_epoch_{self.epoch}.jsonl')
        with open(json_path, 'w') as f:
            for item in to_save:
                f.write(json.dumps(item) + '\n')
        self.epoch += 1
            