from __future__ import annotations
from typing import *
import pandas as pd
from controversy import get_text_controversy


def _get_first_annotations(
    annotations_df: pd.DataFrame,
    rule: Callable[[int, pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame: 
    """
    Helper functions that handle assignment of first annotations for all users

    Args:
        - annotations_df: pd.DataFrame
            dataframe with annotations, represented by text_id, annotator_id and annotation score
        - rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assign first annotation for user
    Returns:
        - pd.DataFrame
            altered annotations dataframe
    """
    for user in pd.unique(annotations_df['annotator_id']):
        choice = rule(user, annotations_df)
        annotations_df['user_annotation_order'].iloc[choice] = 0
    return annotations_df


def _get_next_annotations(
    annotations_df: pd.DataFrame,
    rule: Callable[[int, pd.DataFrame], pd.DataFrame],
    max_annotations_per_user: int
) -> pd.DataFrame:
    """
    Helper functions that handle assignment of next annotations for all users

    Args:
        - annotations_df: pd.DataFrame
            dataframe with annotations, represented by text_id, annotator_id and annotation score
        - rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assign next annotations for user
        - max_annotations_per_user: int
            how many annotations for every user do we want
    Returns:
        - pd.DataFrame
            altered annotations dataframe
    """
    for user in pd.unique(annotations_df['annotator_id']):
        for i in range(1, max_annotations_per_user):
            choice = rule(user, annotations_df)
            annotations_df['user_annotation_order'].iloc[choice] = i
    return annotations_df


def get_annotations(
    data: pd.DataFrame,
    annotations: pd.DataFrame,
    first_annotation_rule: Callable[[int, pd.DataFrame], pd.DataFrame],
    next_annotations_rule: Callable[[int, pd.DataFrame], pd.DataFrame],
    max_annotations_per_user: int,
    annotators_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    '''
    Function that limit annotations number using specified rules

    Args:
        - data: pd.DataFrame 
            dataframe with texts
        - annotations: pd.DataFrame 
            dataframe with annotations, represented by text_id, annotator_id and annotation score
        - annotators_data: Optional[pd.DataFrame]
            dataframe with annotators metadata such as age, gender etc.
            If not None, will be merged with annotations
        - first_annotation_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assignes first annotation to users
        - next_annotations_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assignes every other annotation
        - max_annotations_per_user: int
            how many annotations per user we want to keep after process

    Returns:
        - pd.DataFrame
            altered annotations dataframe

    Raises:
        - AssertionError 
            if max_annotations_per_user <= 0 
    '''  
    assert max_annotations_per_user > 0, f"Max annotations per user must be greater than 0, but got {max_annotations_per_user}"
    # create ID column
    annotations['user_annotation_order'] = -1
    merged_annotations = annotations.merge(data, on='text_id')
    train_annotations, dev_test_annotations = merged_annotations[merged_annotations['split'] == 'train'], merged_annotations[merged_annotations['split'] != 'train']
    
    # take annotation columns
    annotation_columns = annotations.columns
    # possible merge annotators columns
    if annotators_data is not None:
        train_annotations = train_annotations.merge(annotators_data, on='annotator_id')

    # assign first annotations 
    train_annotations = _get_first_annotations(train_annotations, first_annotation_rule)
    # assign next annotations 
    train_annotations = _get_next_annotations(train_annotations, next_annotations_rule, max_annotations_per_user)
    
    # cat annotations
    new_annotations = pd.concat([
        train_annotations[train_annotations['user_annotation_order'] != -1],
        dev_test_annotations
    ])[annotation_columns]
    return new_annotations

def get_controversy(
    data: pd.DataFrame,
    annotations: pd.DataFrame,
    max_annotations_per_user: int
) -> pd.DataFrame:
    '''
    Function that alters a given dataframe by adding a controversy value for every sample using specified rules

    Args:
        - data: pd.DataFrame 
            dataframe with texts
        - annotations: pd.DataFrame 
            dataframe with annotations, represented by text_id, annotator_id and annotation score
        - annotators_data: Optional[pd.DataFrame]
            dataframe with annotators metadata such as age, gender etc.
            If not None, will be merged with annotations
        - first_annotation_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assignes first annotation to users
        - next_annotations_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assignes every other annotation
        - max_annotations_per_user: int
            how many annotations per user we want to keep after process

    Returns:
        - pd.DataFrame
            altered annotations dataframe

    Raises:
        - AssertionError 
            if max_annotations_per_user <= 0 
    ''' 
    annotations_df = data
    annotations_df['controversy'] = -1
    for user in pd.unique(annotations_df['annotator_id']):
        user_annotations = annotations_df[user]
        for i in range(1, max_annotations_per_user):
            user_controversy = annotations_df.filter(annotations_df[user]).get_text_controversy(annotations_df[i])
            # choice = rule(user, annotations_df)
            annotations_df['user_annotation_order'].iloc[choice] = i
    return annotations_df

def get_weighted_controversy(
data: pd.DataFrame,
    annotations: pd.DataFrame,
    rule: Callable[[int, pd.DataFrame], pd.DataFrame],
    max_annotations_per_user: int
) -> pd.DataFrame:
    '''
    Function that alters a given dataframe by adding a weighted controversy value for every sample using specified rules

    Args:
        - data: pd.DataFrame 
            dataframe with texts
        - annotations: pd.DataFrame 
            dataframe with annotations, represented by text_id, annotator_id and annotation score
        - annotators_data: Optional[pd.DataFrame]
            dataframe with annotators metadata such as age, gender etc.
            If not None, will be merged with annotations
        - first_annotation_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assignes first annotation to users
        - next_annotations_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
            function that assignes every other annotation
        - max_annotations_per_user: int
            how many annotations per user we want to keep after process

    Returns:
        - pd.DataFrame
            altered annotations dataframe

    Raises:
        - AssertionError 
            if max_annotations_per_user <= 0 
    ''' 
    annotations_df = data
    for user in pd.unique(annotations_df['annotator_id']):
        user_annotations = annotations_df[user]
        for i in range(1, max_annotations_per_user):
            annotation_controversy = annotations_df.filter(annotations_df[user])
            annotation_controversy.get_text_controversy(annotations_df[i])
            # choice = rule(user, annotations_df)
            annotations_df['user_annotation_order'].iloc[choice] = i
    return annotations_df

