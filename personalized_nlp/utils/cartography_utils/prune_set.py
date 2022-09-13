from typing import Dict
import os
import re
import glob
import tqdm


import numpy as np
import pandas as pd


def create_folds_dict(root: str, file_ext: str = '.csv', regex: str = '(?<=fold_)([0-9]+)(?=\_)') -> Dict[int, str]:
    paths_dict: Dict[int, str] = dict()
    for path in glob.glob(os.path.join(root, f'*{file_ext}')):
        fold = int(re.search(regex, path).group())
        paths_dict[fold] = path
    return paths_dict
    
    
def filter_annotations(metrics_df: pd.DataFrame, sort_by: str, ascending: bool, top_perc: float):
    if sort_by == 'random':
        idx = metrics_df.index.values
        #np.random.shuffle(idx)
        #print(idx)
        return metrics_df.loc[idx, :].head(n=int(len(metrics_df) * top_perc))
    sorted_scores = metrics_df.sort_values(by=[sort_by], ascending=ascending)
    
    selected = sorted_scores.head(n=int(len(sorted_scores) * top_perc))
    return selected

def filter_by_users(metrics_df: pd.DataFrame, sort_by: str, ascending: bool, top_perc: float):
    dfs = []
    metrics_df[['text_id', 'annotator_id']] = metrics_df['guid'].str.split('_', expand=True)
    
    selected = metrics_df.groupby('annotator_id').apply(lambda x: filter_annotations(x, sort_by=sort_by, ascending=ascending, top_perc=top_perc))
    selected.reset_index(drop=True)
    return selected


def prune_train_set(
    original_df: pd.DataFrame, 
    metrics_dict: Dict[int, str], 
    fold_num: int,
    sort_by: str, 
    ascending: bool, 
    top_perc: float,
    stratify_by_users: bool = False) -> pd.DataFrame:
    
    df = original_df.copy()
    metric_df = pd.read_csv(metrics_dict[fold_num])
    assert sort_by in list(metric_df.columns) + ['random'], f'Sort by must be one of {list(metric_df.columns) + ["random"]}, but got {sort_by}.'
    if stratify_by_users:
        selected = filter_by_users(metric_df, sort_by, ascending, top_perc)
    else:
        selected = filter_annotations(metric_df, sort_by, ascending, top_perc)
    
    df['guid'] = df['text_id'].astype(str).str.cat(df['annotator_id'].astype(str), sep='_')
    df_selected = df[df['guid'].isin(selected['guid'])]
    return df_selected
    


if __name__ == '__main__':
    path = create_folds_dict('/home/konradkaranowski/storage/personalized-nlp/storage/outputs/cartography_outputs/cartography_wiki_agr_model=onehot/metrics/class_aggression')
