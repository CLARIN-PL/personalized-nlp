import pandas as pd
import numpy as np

from active_learning.algorithms.base import TextSelectorBase

from typing import Dict


class TextScaledAnnotationDiversitySelector(TextSelectorBase):

    def __init__(self, class_dims=None) -> None:
        """Selector basing on the number of already existing text annotations. 
        If there is no annoations for a specific text then the measure will return 1. 
        Otherwise it will return 1/(1+number_of_annotations)
        """
        super().__init__(class_dims)

        self.text_ids_cycle = None

    def get_scaled_annotation_diversity(
            df: pd.DataFrame, annotated_text_ids: Dict[str,
                                                       pd.DataFrame]) -> float:
        text_id = df['text_id'].unique().tolist()[0]

        if text_id not in annotated_text_ids.keys():
            return 1.0
        else:
            return 1.0 / (1.0 + len(annotated_text_ids[text_id]))

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
        annotated_text_ids = annotated.groupby("text_id").groups

        original_columns = not_annotated.columns
        view = not_annotated.loc[:, :]
        view.loc[:,
                 'scaled_annotations_diversity'] = view.loc[:, 'text_id'].map(
                     view.groupby('text_id').apply(
                         lambda x: self.get_scaled_annotation_diversity(
                             x, annotated_text_ids)))

        return_df = view.sort_values(
            by=['scaled_annotations_diversity'],
            ascending=False).iloc[:amount, :].loc[:, original_columns]
        not_annotated.drop(columns=['scaled_annotations_diversity'],
                           inplace=True)

        return return_df