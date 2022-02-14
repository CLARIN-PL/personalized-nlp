from pstats import Stats
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

# the metrics

# def _get_first_annotations(
#     annotations_df: pd.DataFrame,
#     rule: Callable[[int, pd.DataFrame], pd.DataFrame]
# ) -> pd.DataFrame: 
#     """
#     Helper functions that handle assignment of first annotations for all users

#     Args:
#         - annotations_df: pd.DataFrame
#             dataframe with annotations, represented by text_id, annotator_id and annotation score
#         - rule: Callable[[int, pd.DataFrame], pd.DataFrame]
#             function that assign first annotation for user
#     Returns:
#         - pd.DataFrame
#             altered annotations dataframe
#     """
#     for user in pd.unique(annotations_df['annotator_id']):
#         choice = rule(user, annotations_df)
#         annotations_df['user_annotation_order'].iloc[choice] = 0
#     return annotations_df


# def _get_next_annotations(
#     annotations_df: pd.DataFrame,
#     rule: Callable[[int, pd.DataFrame], pd.DataFrame],
#     max_annotations_per_user: int
# ) -> pd.DataFrame:
#     """
#     Helper functions that handle assignment of next annotations for all users

#     Args:
#         - annotations_df: pd.DataFrame
#             dataframe with annotations, represented by text_id, annotator_id and annotation score
#         - rule: Callable[[int, pd.DataFrame], pd.DataFrame]
#             function that assign next annotations for user
#         - max_annotations_per_user: int
#             how many annotations for every user do we want
#     Returns:
#         - pd.DataFrame
#             altered annotations dataframe
#     """
#     for user in pd.unique(annotations_df['annotator_id']):
#         for i in range(1, max_annotations_per_user):
#             choice = rule(user, annotations_df)
#             annotations_df['user_annotation_order'].iloc[choice] = i
#     return annotations_df


# def get_annotations(
#     data: pd.DataFrame,
#     annotations: pd.DataFrame,
#     first_annotation_rule: Callable[[int, pd.DataFrame], pd.DataFrame],
#     next_annotations_rule: Callable[[int, pd.DataFrame], pd.DataFrame],
#     max_annotations_per_user: int,
#     annotators_data: Optional[pd.DataFrame] = None,
# ) -> pd.DataFrame:
#     '''
#     Function that limit annotations number using specified rules

#     Args:
#         - data: pd.DataFrame 
#             dataframe with texts
#         - annotations: pd.DataFrame 
#             dataframe with annotations, represented by text_id, annotator_id and annotation score
#         - annotators_data: Optional[pd.DataFrame]
#             dataframe with annotators metadata such as age, gender etc.
#             If not None, will be merged with annotations
#         - first_annotation_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
#             function that assignes first annotation to users
#         - next_annotations_rule: Callable[[int, pd.DataFrame], pd.DataFrame]
#             function that assignes every other annotation
#         - max_annotations_per_user: int
#             how many annotations per user we want to keep after process

#     Returns:
#         - pd.DataFrame
#             altered annotations dataframe

#     Raises:
#         - AssertionError 
#             if max_annotations_per_user <= 0 
#     '''  
#     assert max_annotations_per_user > 0, f"Max annotations per user must be greater than 0, but got {max_annotations_per_user}"
#     # create ID column
#     annotations['user_annotation_order'] = -1
#     merged_annotations = annotations.merge(data, on='text_id')
#     train_annotations, dev_test_annotations = merged_annotations[merged_annotations['split'] == 'train'], merged_annotations[merged_annotations['split'] != 'train']
    
#     # take annotation columns
#     annotation_columns = annotations.columns
#     # possible merge annotators columns
#     if annotators_data is not None:
#         train_annotations = train_annotations.merge(annotators_data, on='annotator_id')

#     # assign first annotations 
#     train_annotations = _get_first_annotations(train_annotations, first_annotation_rule)
#     # assign next annotations 
#     train_annotations = _get_next_annotations(train_annotations, next_annotations_rule, max_annotations_per_user)
    
#     # cat annotations
#     new_annotations = pd.concat([
#         train_annotations[train_annotations['user_annotation_order'] != -1],
#         dev_test_annotations
#     ])[annotation_columns]
#     return new_annotations

def num_of_annotations(data: pd.DataFrame):
    return data.shape[0]


def var_ratio(self, data: pd.DataFrame):
# def variation_ratio(self, pred_matrix):
    #  """Computes and returns the variation ratios of the predictions in the given prediction matrix"""
    preds = [preds.argmax(1) for preds in pd]
    mode = Stats.mode(preds, axis=1)
    return 1 - (mode[1].squeeze() / self.T)

def get_entropy(annotations: pd.pd.DataFrame, annotation_columns: List[str], mean=False):
    def _entropy(labels, base=None):
        _, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)

    return _get_text_controversy(annotations, annotation_columns, _entropy, mean)

def _get_text_controversy(annotations: pd.DataFrame, annotation_columns: List[str], method: Callable, mean: bool):
    texts_controversy_df = annotations.loc[:, ['text_id']].drop_duplicates().reset_index(drop=True)
    if isinstance(annotation_columns, str):
        annotation_columns = [annotation_columns]
    controversy_columns = [col + '_controversy' for col in annotation_columns]
    for annotation_col, controversy_col in zip(annotation_columns, controversy_columns):
        text_controversy_dict = annotations.groupby('text_id')[annotation_col].apply(method).to_dict()
        texts_controversy_df[controversy_col] = texts_controversy_df.text_id.apply(text_controversy_dict.get)

    if mean:
        texts_controversy_df['mean_controversy'] = texts_controversy_df.loc[:, controversy_columns].mean(axis=1)
        texts_controversy_df = texts_controversy_df.loc[:, ['text_id', 'mean_controversy']]

    return texts_controversy_df

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


def get_conformity(self, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        if annotations is None:
            annotations = self.annotations

        df = annotations.copy()
        column = self.annotation_column

        mean_score = df.groupby("text_id").agg(score_mean=(column, "mean"))
        df = df.merge(mean_score.reset_index())

        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)

        df["is_major_vote"] = df["text_major_vote"] == df[column]
        df["is_major_vote"] = df["is_major_vote"].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        conformity_df = df.groupby("annotator_id").agg(
            conformity=("is_major_vote", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_conformity=("is_major_vote", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_conformity=("is_major_vote", "mean")
        )

        return conformity_df

def get_weighted_conformity(self, annotations: pd.DataFrame = None) -> pd.DataFrame:
        """Computes conformity for each annotator. Works only for binary classification problems."""
        if annotations is None:
            annotations = self.annotations

        df = annotations.copy()
        column = self.annotation_column

        mean_score = df.groupby("text_id").agg(score_mean=(column, "mean"))
        df = df.merge(mean_score.reset_index())

        df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)

        df["is_major_vote"] = df["text_major_vote"] == df[column]
        df["is_major_vote"] = df["is_major_vote"].astype(int)

        positive_df = df[df.text_major_vote == 1]
        negative_df = df[df.text_major_vote == 0]

        conformity_df = df.groupby("annotator_id").agg(
            conformity=("is_major_vote", "mean")
        )
        conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
            pos_conformity=("is_major_vote", "mean")
        )
        conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
            neg_conformity=("is_major_vote", "mean")
        )

        return conformity_df
