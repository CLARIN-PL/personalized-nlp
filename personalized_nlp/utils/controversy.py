import pandas as pd

import numpy as np
from scipy.stats import entropy


def _entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def get_text_controversy(annotations: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return annotations.groupby('text_id')['aggression'].apply(_entropy)
