from typing import Any, Optional
import warnings
import json
import os
from pytorch_lightning.callbacks import Callback

from settings import CARTOGRAPHY_OUTPUTS_DIR_NAME



class CartographySaveCallback(Callback):
    
    def __init__(
            self, 
            dir_name: str = 'cartography',
            fold_num: int = 0
        ) -> None:
        """Callback for generating training dynamics. At this moment support ONLY:
        - CLASSIFICATION
        - MULTILABEL 

        Args:
            dir_name (str, optional): 
                Name for dir with training dynamics.
        """
        super(CartographySaveCallback, self).__init__()
        self.outputs = []
        self.epoch = 0
        
        self.save_dir = CARTOGRAPHY_OUTPUTS_DIR_NAME / dir_name
        self.fold_num = fold_num
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            warnings.warn(f'Dir {self.save_dir} is not empty! This experiment will override existing logs!')

    
    def on_train_batch_end(self, 
                           trainer: "pl.Trainer", 
                           pl_module: "pl.LightningModule", 
                           outputs: dict, 
                           batch: int, 
                           batch_idx: int, 
                           unused: int = 0
        ) -> None:
        self.outputs.append(outputs)
        
    def on_train_epoch_end(self, *args, **kwargs) -> None:
        json_path = os.path.join(self.save_dir, f'dynamics_epoch_{self.epoch}.jsonl')
        with open(json_path, 'w') as f:
            for suboutput in self.outputs:
                for text_id, user_id, logits, y_true in zip(
                    suboutput['x']['text_ids'],
                    suboutput['x']['annotator_ids'],
                    suboutput['output'],
                    suboutput['y'],
                ): 
                    to_save = {
                        'guid': f'text={text_id}_user={user_id}',
                        f'logits_epoch_{self.epoch}': logits.tolist(),
                        'gold': y_true.int().tolist(),
                        'class_names': suboutput['class_names']
                    }
                    f.write(json.dumps(to_save) + '\n')
        self.epoch += 1
        self.outputs = []
