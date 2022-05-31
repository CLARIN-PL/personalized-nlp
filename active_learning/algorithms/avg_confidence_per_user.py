import warnings

import pandas as pd
import numpy as np

from active_learning.algorithms.base import TextSelectorBase


class AverageConfidencePerUserSelector(TextSelectorBase):
    
    def __init__(self, select_minimal_texts: bool = True, *args, **kwargs) -> None:
        """Selector basing on model's average confidence on text. If there is a conflict, it is resolved based on mode's average confidence on user.
        
        Args:
            select_minimal_texts (bool, optional): Wheter to choose annotations with minimal (True) or maximal (False) confidence. Defaults to True.
        """
        super(AverageConfidencePerUserSelector, self).__init__()
        self.select_minimal_texts: bool = select_minimal_texts
    
    def select_annotations(
            self,
            texts: pd.DataFrame,
            amount: int,
            annotated: pd.DataFrame,
            not_annotated: pd.DataFrame,
            confidences: np.ndarray,
        ) -> pd.DataFrame:
            """Select annotations using rule, described in __init__()

            Args:
                texts (pd.DataFrame): Dataframe with texts data. It contains (at least)
                two columns: `text_id` (int) and `text` (str)
                amount (int): Number of annotations to select from `not_annotated` dataframe.
                annotated (pd.DataFrame): Already selected annotations. It is a Dataframe with
                (at least) two columns: `text_id` and `annotator_id`.
                not_annotated (pd.DataFrame): Annotations to select. It is a Dataframe with
                (at least) two columns: `text_id` and `annotator_id`. The algorithm should return
                `amount` of rows from this dataframe.
                confidences (np.ndarray): Numpy array of prediction confidences for all annotations
                in `not_annotated` dataframe. It has length equal to length of `not_annotated` dataframe.

            Returns:
                pd.DataFrame: Dataframe with subset of rows of `not_annotated` dataframe with length equal to
                `amount`.
            """
            if confidences is not None:
                original_columns = not_annotated.columns
                view = not_annotated.loc[:, :]
                view.loc[:, 'max_confidences'] = confidences.max(axis=1)
                view.loc[:, 'text_avg'] = view.loc[:, 'text_id'].map(view.groupby('text_id')['max_confidences'].mean())
                view.loc[:, 'ann_avg'] = view.loc[:, 'annotator_id'].map(view.groupby('annotator_id')['max_confidences'].mean())
                
                return_df = view.sort_values(by=['text_avg', 'ann_avg'], ascending=self.select_minimal_texts).iloc[:amount, :].loc[:, original_columns]
                not_annotated.drop(columns=['max_confidences', 'text_avg', 'ann_avg'], inplace=True)
                
                return return_df
            warnings.warn(f'There is no confidences, sampled {amount} of samples from not annotated data.')

            return not_annotated.sample(n=amount)
