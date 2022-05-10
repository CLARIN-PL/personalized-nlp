from typing import List
import pandas as pd
import numpy as np


def random_selector(
    texts: pd.DataFrame,
    amount: int,
    annotated: pd.DataFrame,
    not_annotated: pd.DataFrame,
    confidences: np.ndarray,
):
    return not_annotated.sample(n=amount)
