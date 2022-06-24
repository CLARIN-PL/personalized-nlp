from typing import Dict, Tuple

import pandas as pd


def get_annotator_id_map(
        annotations_df: pd.DataFrame, 
        annotator_col: str
    ) -> Dict[int, int]:
    """Creates a map for original annotator id to new annotator id.

    Args:
        annotations_df (pd.DataFrame): Dataframe containing annotations.
        annotator_col (str): Name of annotator id column.

    Returns:
        Dict[int, int]: Map: [old_id1, old_id2, ..., old_idN] -> [1, 2, ..., N]
    """
    annotator_id_category = annotations_df[annotator_col].astype("category")
    annotator_id_idx_dict = {
        a_id: idx for idx, a_id in enumerate(annotator_id_category.cat.categories)
    }
    return annotator_id_idx_dict


def get_text_id_map(
        text_df: pd.DataFrame, 
        text_col: str
    ) -> Dict[int, int]:
    """Creates a map for original text id to new text id.
    New indexing is required for embeddings to work.

    Args:
        text_df (pd.DataFrame): Dataframe containing texts.
        text_col (str): Name of text id column.

    Returns:
        Dict[int, int]: Map: [old_id1, old_id2, ..., old_idN] -> [1, 2, ..., N]
    """
    text_id_idx_dict = (
                text_df.loc[:, [text_col]]
                .reset_index()
                .set_index(text_col)
                .to_dict()["index"]
            )
    return text_id_idx_dict


def reindex_texts_and_annotations(
    annotations_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    text_col: str,
    annotator_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reindexes texts and annotations using bijection [x1, x2, x3, ...] -> [1, 2, 3, ...]

    Args:
        annotations_df (pd.DataFrame): Dataframe with annotations.
        texts_df (pd.DataFrame): Dataframe with texts.
    """
    # keep original indexes
    annotations_df[f'{text_col}_original'] = annotations_df[text_col]
    annotations_df[f'{annotator_col}_original'] = annotations_df[annotator_col]
    texts_df[f'{text_col}_original'] = texts_df[text_col]
    
    annotator_id_map = get_annotator_id_map(annotations_df, annotator_col)
    text_id_map = get_text_id_map(texts_df, text_col)
    
    texts_df[text_col] = texts_df[text_col].replace(text_id_map)
    annotations_df[annotator_col] = annotations_df[annotator_col].replace(annotator_id_map)
    annotations_df[text_col] = annotations_df[text_col].replace(text_id_map)
    
    return annotations_df, texts_df
