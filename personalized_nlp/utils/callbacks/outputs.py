from typing import *
import abc
import os
from datetime import datetime

import wandb
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import Callback

from personalized_nlp.settings import STORAGE_DIR


class AbstractSaveOutputsCallback(Callback):

    def __init__(self, save_text: bool) -> None:
        """Saves the predictions to the .csv file. Csv file is stored in:
        ``` logs/wandb/{wandb.run.dir}/files/{save_name}``` 
        Stored attributes: [logit0 ..., logitN-1, y_true0, ... , y_trueK-1, annotator_id, text_id, ?text]

        Args:
            save_name (str, optional): Name of csv file. Defaults to 'outputs.csv'.
        """
        super(AbstractSaveOutputsCallback, self).__init__()
        self.outputs = []
        self.save_text = save_text

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.outputs.append(outputs)

    # def _create_logits(self, is_reggression: bool, p_true) -> Dict[str, np.ndarray]:
    #     pass

    def _create_dataframe(self) -> pd.DataFrame:
        if not self.outputs:
            return None

        dfs = []
        for suboutput in self.outputs:
            x = suboutput['x']
            y_true = suboutput['y']
            y_pred = suboutput['y_pred']
            metric_dict = {
                f'y_pred_{i}': y_pred[:, i].cpu().reshape(-1).numpy() for i in range(y_pred.shape[1])
            }
            metric_dict['text_ids'] = x['text_ids'].cpu().numpy()
            metric_dict['annotator_ids'] = x['annotator_ids'].cpu().numpy()
            metric_dict['y_true'] = y_true.cpu().reshape(-1).numpy()
            if self.save_text:
                metric_dict['raw_texts'] = x['raw_texts']
            df = pd.DataFrame(metric_dict)
            dfs.append(df)
        cat_df = pd.concat(dfs, ignore_index=True)     
        return cat_df   

    @abc.abstractmethod
    def on_test_end(self, *args, **kwargs):
        pass



class SaveOutputsWandb(AbstractSaveOutputsCallback):

    def __init__(self, save_name: str = 'outputs.csv', save_text: bool = True):
        super(SaveOutputsWandb, self).__init__(save_text)
        self.save_name = save_name

    def on_test_end(self, *args, **kwargs) -> None:
        df = self._create_dataframe()
        if df is not None:
            df.to_csv(os.path.join(wandb.run.dir, self.save_name), index=False)
        # TODO warning


class SaveOutputsLocal(AbstractSaveOutputsCallback):
    
    def __init__(self, save_dir: str, save_text: bool = True, **kwargs) -> None:
        super(SaveOutputsLocal, self).__init__(save_text)
        self.save_dir = os.path.join(
            STORAGE_DIR,
            save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_name = f"{datetime.now().strftime('%m%d%Y_%h%m%s')}" + '_'.join(f'{key}={value}' for key, value in kwargs.items()) + ".csv"

    def on_test_end(self, *args, **kwargs):
        df = self._create_dataframe()
        if df is not None:
            df.to_csv(os.path.join(wandb.run.dir, self.save_name), index=False)
        save_path = os.path.join(self.save_dir, self.save_name)
        df.to_csv(save_path, index=False)