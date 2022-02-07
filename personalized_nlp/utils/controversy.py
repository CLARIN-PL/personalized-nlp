from cgitb import text
import pandas as pd

import numpy as np
from scipy.stats import entropy

from typing import Callable, List


def get_texts_entropy(annotations: pd.DataFrame, annotation_columns: List[str], mean=False):
    """ Calculate entropy of text annotations.

    Args:
        annotations (pd.DataFrame): Dataframe with text annotations. It has to contain 'text_id' column.
        annotation_columns (str): Columns of annotations dataframe for which the entropy will be calculated
        mean (bool): If true, the entropy will be averaged over all columns.
    """
    def _entropy(labels, base=None):
        _, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)
    
    return _get_text_controversy(annotations, annotation_columns, _entropy, mean)

def get_texts_std(annotations: pd.DataFrame, annotation_columns: List[str], mean=False):
    """ Calculate std of text annotations.

    Args:
        annotations (pd.DataFrame): Dataframe with text annotations. It has to contain 'text_id' column.
        annotation_columns (str): Columns of annotations dataframe for which the std will be calculated
        mean (bool): If true, the entropy will be averaged over all columns.
    """
    return _get_text_controversy(annotations, annotation_columns, np.std, mean)

def _get_text_controversy(annotations: pd.DataFrame, annotation_columns: List[str], method: Callable, mean: bool):
    texts_controversy_df = annotations.loc[:, ['text_id']].drop_duplicates().reset_index(drop=True)
    
    if isinstance(annotation_columns, str):
        annotation_columns = [annotation_columns]
    controversy_columns = [col + '_controversy' for col in annotation_columns]
    for annotation_col, controversy_col in zip(annotation_columns, controversy_columns):
        text_controversy_dict = annotations.groupby('text_id')[annotation_col].apply(method).to_dict()
        texts_controversy_df[controversy_col] = texts_controversy_df.text_id.apply(text_controversy_dict.get)

    if mean:
        texts_controversy_df['mean_controversy'] = texts_controversy_df.loc[:, controversy_columns].mean(axis=1)
        texts_controversy_df = texts_controversy_df.loc[:, ['text_id', 'mean_controversy']]
        
    return texts_controversy_df