from typing import *

import tqdm
import pandas as pd
import numpy as np


def simulate_annotation_order(user_annotations: pd.DataFrame, user_text_ratios: pd.DataFrame, center_of_weight_method: Callable[[Iterable], float] = np.mean) -> pd.DataFrame:
    
    def _distance(x: Iterable[float], user_ratios: pd.DataFrame) -> float:
        distances = np.abs(user_ratios - center_of_weight_method(x))
        return distances
    
    # set marked
    user_text_ratios['marked'] = False
    
    # get first point
    first_text_id = user_annotations[user_annotations['user_annotation_order'] == 0]['text_id']
    first_text_ratio = user_text_ratios.loc[first_text_id, 'ratio']

    # already annotated texts' ratios
    already_annotated_ratios = [first_text_ratio.item()]
    
    # mark it 
    user_text_ratios.loc[first_text_id, 'marked'] = True
    
    # create result , emulate bulk insert into
    annotation_order = {first_text_id.item(): 0}
    
    # update it while ...
    i = 1
    while i < len(user_annotations):
        # calculate distances
        distances = _distance(
            already_annotated_ratios,
            user_text_ratios[~user_text_ratios['marked']]['ratio']
        )
        # get index of next text
        next_text_id = distances.idxmax()
        
        # append ratio of the text to the list
        already_annotated_ratios.append(
            user_text_ratios.loc[next_text_id, 'ratio']
        )
        
        # append this to annotation order dict
        annotation_order[next_text_id] = i

        # mark this as annotated
        user_text_ratios.loc[next_text_id, 'marked'] = True
        
        i += 1
    # bulk collect
    user_annotations['user_annotation_order'] = user_annotations['text_id'].map(annotation_order)
    return user_annotations


def get_texts_ratios(annotations: pd.DataFrame, column_name: str):
    def _ratio(x: pd.Series) -> float:
        return np.mean(x)
    
    ratios = annotations.groupby('text_id')[column_name].agg(_ratio).to_frame().rename(columns={column_name: 'ratio'}, inplace=False)
    return ratios


def measure_annotation_distance(
    column_name: str, 
    data: pd.DataFrame, 
    max_annotations_per_user: Optional[int] = None,
    center_of_weight_method: Callable[[Iterable], float] = np.mean,
) -> pd.DataFrame:
    ratios = get_texts_ratios(data, column_name=column_name)
    new_annotations = []
    for user in tqdm.tqdm(pd.unique(data['annotator_id'])):
        user_annotations = data[data['annotator_id'] == user]
        user_text_ratios = ratios[ratios.index.isin(user_annotations['text_id'])]
        
        annotations_with_order = simulate_annotation_order(
            user_annotations=user_annotations,
            user_text_ratios=user_text_ratios,
            center_of_weight_method=center_of_weight_method
        )
        new_annotations.append(annotations_with_order)

    annotations = pd.concat(new_annotations, ignore_index=True)
    return annotations