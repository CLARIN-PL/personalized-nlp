import numpy as np
import pandas as pd


def assign_folds(
    annotations_df: pd.DataFrame,
    stratify_by: str,
    num_folds: int = 10
    ) -> pd.DataFrame:
    """Assign folds stratified by specified columns

    Args:
        annotations_df (pd.DataFrame): dataframe with annotations
        stratify_by (str): column to stratify
        num_folds (int, optional): number of folds to create. Defaults to 10.

    Returns:
        pd.DataFrame: _description_
    """
    ids = annotations_df[stratify_by].unique()
    np.random.shuffle(ids)

    folded_ids = np.array_split(ids, num_folds)

    annotations_df["fold"] = 0
    for i in range(num_folds):
        annotations_df.loc[
            annotations_df[stratify_by].isin(folded_ids[i]), "fold"
        ] = i
    return annotations_df
