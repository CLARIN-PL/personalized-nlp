from typing import Dict, Union, List

import numpy as np
import pandas as pd

from scripts.cartography.plot_cartography import compute_metrics


def get_cartography(training_dynamics: Dict[str, Dict[str, Union[List[np.ndarray], int]]]) -> pd.DataFrame:
    num_epochs = len(list(training_dynamics.values())[0]['logits'])
    metrics = compute_metrics(training_dynamics=training_dynamics, num_epochs=num_epochs)
    metrics[['text_id', 'annotator_id']] = metrics['guid'].str.split('_', expand=True)
    metrics['text_id'] = metrics['text_id'].astype(int)
    metrics['annotator_id'] = metrics['annotator_id'].astype(int)
    return metrics