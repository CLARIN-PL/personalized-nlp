from typing import *
import abc
import os
from datetime import datetime

import wandb
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import Callback

from personalized_nlp.settings import STORAGE_DIR, OUTPUTS_DIR_NAME


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

    def _create_outputs(self, is_reggression: bool, x: Sequence[Any], class_names: Sequence[str], base: str) -> Dict[str, np.ndarray]:
        if is_reggression:
            assert len(class_names) == x.shape[1], f'Save output callbacks can only accept reggression, if len(class names) == output shape, but got len(class names) = {len(class_names)} and output shape = {x.shape[1]}'
            out_dict = {
                f'{base}_{class_name}': x[:, i].cpu().reshape(-1).numpy() for i, class_name in enumerate(class_names)
            }
            return out_dict
        else:
            if len(class_names) == 1 and x.shape[1] == 2:
                out_dict = {
                    f'{base}_{class_names[0]}_{i}': x[:, i].cpu().reshape(-1).numpy() for i in range(x.shape[1])
                }
            elif len(class_names) == x.shape[1]:
                out_dict = {
                    f'{base}_{class_name}': x[:, i].cpu().reshape(-1).numpy() for i, class_name in enumerate(class_names)
                }
            else:
                out_dict = {
                    f'{base}_{i}': x[:, i].cpu().reshape(-1).numpy() for i in range(x.shape[1])
                }
            return out_dict


    def _create_dataframe(self) -> pd.DataFrame:
        if not self.outputs:
            return None

        dfs = []
        for suboutput in self.outputs:
            x = suboutput['x']
            y_true = suboutput['y']
            y_pred = suboutput['y_pred']
            is_reggression = suboutput['is_regression']
            class_names = suboutput['class_names']

            y_pred_dict = self._create_outputs(
                is_reggression=is_reggression,
                x=y_pred,
                class_names=class_names,
                base='y_pred'
            )
            y_true_dict = self._create_outputs(
                is_reggression=is_reggression,
                x=y_true,
                class_names=class_names,
                base='y_true'
            )
            metric_dict = {**y_pred_dict, **y_true_dict}

            metric_dict['text_ids'] = x['text_ids'].cpu().numpy()
            metric_dict['annotator_ids'] = x['annotator_ids'].cpu().numpy()

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
        """Callback for loging outputs in local dir. Outputs in .csv file will be saved in $GLOBAL_OUTPUT_PATH/{save_dir},
        and will be named: $TIME_{kwargs.key[0]}={kwargs.value[0]}_{kwargs.key[1]}={kwargs.value[1]}_..._{kwargs.key[N-1]}={kwargs.value[N-1]}
        
        For example:
        `SaveOutputsLocal('mysave_dir', experiment='myexperiment', fold_num=0)`
        will yield this file:
        $GLOBAL_OUTPUT_PATH/mysave_dir/$TIME_experiment=myexperiment_fold_num=0.csv
        Args:
            save_dir (str): _description_
            save_text (bool, optional): Whether to save texts or not. Defaults to True.
        """
        super(SaveOutputsLocal, self).__init__(save_text)
        self.save_dir = os.path.join(
            STORAGE_DIR,
            OUTPUTS_DIR_NAME,
            save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_name = f"{datetime.now().strftime('%m%d%Y_%h%m%s')}" + '_'.join(f'{key}={value}' for key, value in kwargs.items()) + ".csv"

    def on_test_end(self, *args, **kwargs):
        df = self._create_dataframe()
        if df is not None:
            save_path = os.path.join(self.save_dir, self.save_name)
            df.to_csv(save_path, index=False)