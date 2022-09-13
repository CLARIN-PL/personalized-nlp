from typing import *

import numpy as np
import pandas as pd


def get_seed(df: pd.DataFrame, test_fold: int, val_fold: int, perc: float) -> pd.DataFrame:
    df['guid'] = df['rev_id'].astype(str).str.cat(df['worker_id'].astype(str), sep='_')
    df_train, df_val, df_test = df[~df['fold'].isin([test_fold, val_fold])], df[df['fold'] == val_fold], df[df['fold'] == test_fold]
    df_val.loc[:, 'available'] = True
    df_test.loc[:, 'available'] = True
    
    seed_size = int(len(df_train) * perc)
    indicies = [True] * seed_size + [False] * (len(df_train) - seed_size)
    np.random.shuffle(indicies)
    df_train.loc[:, 'available'] = indicies
    return pd.concat([df_train, df_test, df_val], ignore_index=True)