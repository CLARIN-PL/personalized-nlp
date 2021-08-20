from typing import List

import pandas as pd

def get_annotator_biases(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    text_means = df.groupby('text_id').mean().loc[:, columns]
    text_stds = df.groupby('text_id').std().loc[:, columns]
    
    df = df.join(text_means, rsuffix='_mean', on='text_id').join(text_stds, rsuffix='_std', on='text_id')
    
    for col in columns:
        df[col + '_z_score'] = (df[col] - df[col + '_mean']) / df[col + '_std']
    
    annotator_biases = df.groupby('annotator_id').mean().loc[:, [col + '_z_score' for col in columns]]

    annotator_biases.columns = [col + '_bias' for col in columns]
    
    return annotator_biases