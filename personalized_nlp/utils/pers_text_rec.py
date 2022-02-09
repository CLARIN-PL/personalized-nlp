from typing import *

import pandas as pd
import numpy as np



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
    merged_annotations = next_annotations_rule(merged_annotations, max_annotations_per_user)

    annotations_with_id = merged_annotations[annotations_column]
    return annotations_with_id
