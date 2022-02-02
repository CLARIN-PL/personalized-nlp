from typing import *

import pandas as pd
import numpy as np


# def _get_first_annotations(
#     annotations_df: pd.DataFrame,
#     rule: Callable
# ) -> pd.DataFrame:
#     for user in pd.unique(annotations_df.annotator_id):
#         annotations_df = rule(user, annotations_df)
#         # annotations_df['user_annotation_order'].iloc[choice] = 0
#     return annotations_df


# def _get_next_annotations(
#     annotations_df: pd.DataFrame,
#     rule: Callable,
#     max_annotations_per_user: int
# ) -> pd.DataFrame:
#     availible_annotations_for_users = {
#         k: min(v, max_annotations_per_user - 1) 
#             for k, v in dict(annotations_df.groupby('annotator_id').count()['text_id']).items()
#         }
#     for user in pd.unique(annotations_df['annotator_id']):
#         for i in range(1, availible_annotations_for_users[user]):
#             annotations_df = rule(user, annotations_df)
#             # annotations_df['user_annotation_order'].iloc[choice] = i
#     return annotations_df


def assign_annotations(
    data: pd.DataFrame,
    annotations: pd.DataFrame,
    first_annotation_rule: Callable,
    next_annotations_rule: Callable,
    max_annotations_per_user: Optional[int] = None
) -> pd.DataFrame:
    if max_annotations_per_user is None:
        max_annotations_per_user = len(annotations())
    assert max_annotations_per_user > 0, f"Max annotations per user must be grater than 0 but got {max_annotations_per_user}"

    # create ID column
    annotations['user_annotation_order'] = -1
    annotations_column = annotations.columns

    # merge text and annotations
    merged_annotations = annotations.merge(data['text_id'], on='text_id')

    # create first annotations
    merged_annotations = first_annotation_rule(merged_annotations)

    # create second annotations
    merged_annotations = next_annotations_rule(merged_annotations)

    annotations_with_id = merged_annotations[annotations_column]
    return annotations_with_id
