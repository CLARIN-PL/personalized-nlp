from typing import * 


import pandas as pd
import numpy as np


# TODO correct this!
def set_first_assignemnt(
        column_name,
        data: pd.DataFrame,
        max_annotations_per_user: Optional[int] = None,
        **kwargs) -> pd.DataFrame:
    data['user_annotation_order'] = np.random.permutation(len(data.index))
    data['user_annotation_order'] = data.groupby('annotator_id')['user_annotation_order'].apply(lambda x: np.argsort(x))
    data['user_annotation_order'][data['user_annotation_order'] != 0] = -1
    return data


def random_assignment(
        column_name: str,
        data: pd.DataFrame,
        max_annotations_per_user: Optional[int] = None,
        **kwargs) -> pd.DataFrame:
    data['user_annotation_order'] = np.random.permutation(len(data.index))
    data['user_annotation_order'] = data.groupby('annotator_id')['user_annotation_order'].apply(lambda x: np.argsort(x))
    return data