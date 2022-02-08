from typing import *

import pandas as pd
import numpy as np


def assign_annotations(
        data: pd.DataFrame,
        annotations: pd.DataFrame,
        first_annotation_rule: Callable,
        next_annotations_rule: Callable,
        max_annotations_per_user: Optional[int] = None) -> pd.DataFrame:
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
    merged_annotations = next_annotations_rule(merged_annotations,
                                               max_annotations_per_user)

    annotations_with_id = merged_annotations[annotations_column]
    return annotations_with_id


def random_assignment(
        merged_annotations: pd.DataFrame,
        max_annotations_per_user: Optional[int] = None) -> pd.DataFrame:
    for user_id in merged_annotations['annotator_id'].unique().tolist():
        num_annotations = len(
            merged_annotations[merged_annotations['annotatior_id'] == user_id])

        annotations_order = np.random.choice(range(0, num_annotations),
                                             size=num_annotations,
                                             replace=False)

        order_column_id = merged_annotations.columns.get_loc(
            "user_annotation_order")

        for idx_ind, row_idx in enumerate(merged_annotations.index[
                merged_annotations['annotator_id'] == user_id].tolist()):
            merged_annotations.iloc[
                row_idx, order_column_id] = annotations_order[idx_ind]

        merged_annotations = merged_annotations.loc[
            merged_annotations['annotator_id'] == user_id,
            'user_annotation_order'] = np.random.choice(range(
                0, num_annotations),
                                                        size=num_annotations,
                                                        replace=False)
