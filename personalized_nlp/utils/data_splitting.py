from typing import Sequence

import numpy as np
import pandas as pd


def split_texts(df: pd.DataFrame, sizes: Sequence[float], split_column_name="text_split") -> pd.DataFrame:
    """Splits texts. Used in user-fold scenario.

    Args:
        df (pd.DataFrame): Dataframe with annotations.
        sizes (Sequence[float]): Sizes of splits.
        split_column_name (str, optional): Name of the text split column. Defaults to "text_split".

    Returns:
        pd.DataFrame: Dataframe splitted. 
    """
    assert sum(sizes) == 1.0, f'Sizes should sum to 1, but sums to {sum(sizes)}'
    present_ratio, past_ratio, future1_ratio, future2_ratio = sizes
    

    present_idx = int(present_ratio * len(df.index))
    past_idx = int(past_ratio * len(df.index)) + present_idx
    future1_idx = int(future1_ratio * len(df.index)) + past_idx

    indexes = np.arange(len(df.index))
    np.random.shuffle(indexes)

    df = df.copy()

    df[split_column_name] = ""
    split_column_name_loc = df.columns.get_loc(split_column_name)
    print(split_column_name)

    df.iloc[indexes[:present_idx], split_column_name_loc] = "present"
    df.iloc[indexes[present_idx:past_idx], split_column_name_loc] = "past"
    df.iloc[indexes[past_idx:future1_idx], split_column_name_loc] = "future1"
    df.iloc[indexes[future1_idx:], split_column_name_loc] = "future2"

    return df
