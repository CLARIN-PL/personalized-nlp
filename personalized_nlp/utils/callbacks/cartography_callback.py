from typing import Any, Dict
import os
import json
import warnings
from pathlib import PosixPath

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from settings import CARTOGRAPHY_OUTPUTS_DIR_NAME, CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME, CARTOGRAPHY_PLOTS_DIR_NAME, CARTOGRAPHY_METRICS_DIR_NAME, CARTOGRAPHY_FILTER_DIR_NAME



class CartographySaveCallback(Callback):

    def __init__(
            self, 
            dir_name: str,
            fold_num: int,
            fold_nums: int
        ) -> None:
        """Callback to save training dynamics used by cartography datamodule

        Args:
            dir_name (str): name of directory with the cartography project. The path will become outputs/$CARTOGRAPHY_OUTPUTS_DIR_NAME/$dir_name.
            fold_num (int): current fold (every fold is stored in separate dir inside $dir_name).
            fold_nums (int): total number of folds.
        """
        super(CartographySaveCallback, self).__init__()
        self.outputs = []
        self.epoch = 0
        
        base_dir = CARTOGRAPHY_OUTPUTS_DIR_NAME / dir_name
        
        self.meta_dict = {
            "project_dir": str(base_dir),
            "fold_nums": fold_nums,
            "train_dir": str(base_dir / CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME),
            "plots_dir": str(base_dir / CARTOGRAPHY_PLOTS_DIR_NAME),
            "metrics_dir": str(base_dir / CARTOGRAPHY_METRICS_DIR_NAME),
            "filter_dir": str(base_dir / CARTOGRAPHY_FILTER_DIR_NAME)
        }
        
        self.save_dir = PosixPath(self.meta_dict["train_dir"]) / f'fold_{fold_num}'
        self.meta_path = PosixPath(self.meta_dict["project_dir"]) / f'meta.json'
        self.fold_num = fold_num
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            warnings.warn(f'Dir {self.save_dir} is not empty! This experiment will override existing logs!')
        self._write_meta()
            
            
    def _write_meta(self) -> None:
        """Writes metadata about the project to the meta.json file.
        """
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(self.meta_dict))

    
    def on_train_batch_end(self, 
                           trainer: pl.Trainer, 
                           pl_module: pl.LightningModule, 
                           outputs: Dict[Any, Any], 
                           batch: int, 
                           batch_idx: int, 
                           unused: int = 0
        ) -> None:
        """Append output at the end of train batch
        """
        self.outputs.append(outputs)
        
    def on_train_epoch_end(self, *args, **kwargs) -> None:
        """Creates training dynamics file after every training epochs.
        """
        json_path = os.path.join(self.save_dir, f'dynamics_epoch_{self.epoch}.jsonl')
        lines = []
        for suboutput in self.outputs:
            for text_id, user_id, logits, y_true in zip(
                suboutput['x']['text_ids'],
                suboutput['x']['annotator_ids'],
                suboutput['output'],
                suboutput['y'],
            ): 
                to_save = {
                    'guid': f'{text_id}_{user_id}',
                    f'logits_epoch_{self.epoch}': logits.tolist(),
                    'gold': y_true.int().tolist(),
                    'class_names': suboutput['class_names']
                }
                lines.append(json.dumps(to_save) + '\n')
        with open(json_path, 'w') as f:
            f.writelines(lines)
        self.epoch += 1
        self.outputs = []
