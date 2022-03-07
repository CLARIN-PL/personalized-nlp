from typing import *

import torch
import pandas as pd
import wandb


def _log_predictions(x: Dict[str, Any], y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    metric_dict = {
        f'y_pred_{i}': y_pred[:, i].cpu().reshape(-1).numpy() for i in range(y_pred.shape[1])
    }
    metric_dict['text_ids'] = x['text_ids'].cpu().numpy()
    metric_dict['annotator_ids'] = x['annotator_ids'].cpu().numpy()
    metric_dict['y_true'] = y_true.cpu().reshape(-1).numpy()
    df = pd.DataFrame(metric_dict)
    table = wandb.Table(dataframe=df)
    wandb.log({'predictions': table})
