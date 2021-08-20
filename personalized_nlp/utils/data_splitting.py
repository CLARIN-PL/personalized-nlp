import numpy as np
import pandas as pd

def split_texts(df, sizes):
    present_ratio, past_ratio, future1_ratio, future2_ratio = sizes
    
    present_idx = int(present_ratio * len(df.index))
    past_idx = int(past_ratio * len(df.index)) + present_idx
    future1_idx = int(future1_ratio * len(df.index)) + past_idx
    
    indexes = np.arange(len(df.index))
    np.random.shuffle(indexes)
    
    df = df.copy()
    df['split'] = ''
    df.iloc[indexes[:present_idx], df.columns.get_loc('split')] = 'present'
    df.iloc[indexes[present_idx:past_idx], df.columns.get_loc('split')] = 'past'
    df.iloc[indexes[past_idx:future1_idx], df.columns.get_loc('split')] = 'future1'
    df.iloc[indexes[future1_idx:], df.columns.get_loc('split')] = 'future2'
    
    return df