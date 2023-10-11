from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def get_text_controversy(annotations: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return annotations.groupby("text_id")[score_column].apply(_entropy)


def get_conformity(
    annotations: pd.DataFrame,
    columns: List[str],
    conformity_type: Optional[str] = "all",
) -> pd.DataFrame:
    conformity_dfs = []
    if conformity_type != "weighted":
        g_conformity = get_general_conformity(annotations, columns)
        conformity_dfs.append(g_conformity)

    if conformity_type != "normal":
        w_conformity = get_weighted_conformity(annotations, columns)
        conformity_dfs.append(w_conformity)

    return pd.concat(conformity_dfs, axis=1)


def get_general_conformity(
    annotations: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """Computes conformity for each annotator. Works only for binary classification problems."""
    df = annotations.copy()

    mean_scores = df.groupby("text_id")[columns].mean()
    mean_scores = mean_scores.rename(columns=lambda col: col + "_mean")
    df = df.merge(mean_scores.reset_index())

    conformity_dfs = []
    for col in columns:
        df[f"{col}_major_vote"] = (df[f"{col}_mean"] > 0.5).astype(int)
        df[f"{col}_is_major_vote"] = df[f"{col}_major_vote"] == df[col]
        df[f"{col}_is_major_vote"] = df[f"{col}_is_major_vote"].astype(int)

        positive_df = df[df[f"{col}_major_vote"] == 1]
        negative_df = df[df[f"{col}_major_vote"] == 0]

        col_conformity_df = (
            df.groupby(["annotator_id"])[[f"{col}_is_major_vote"]]
            .mean()
            .rename(columns=lambda _: col + "_conformity")
            .reset_index()
        )

        pos_col_conformity_df = (
            positive_df.groupby(["annotator_id"])[[f"{col}_is_major_vote"]]
            .mean()
            .rename(columns=lambda _: col + "_pos_conformity")
            .reset_index()
        )
        neg_col_conformity_df = (
            negative_df.groupby(["annotator_id"])[[f"{col}_is_major_vote"]]
            .mean()
            .rename(columns=lambda _: col + "_neg_conformity")
            .reset_index()
        )

        conformity_df = col_conformity_df.merge(pos_col_conformity_df, how="left")
        conformity_df = conformity_df.merge(neg_col_conformity_df, how="left")
        conformity_dfs.append(conformity_df.set_index("annotator_id"))

    return pd.concat(conformity_dfs, axis=1).fillna(1.0)


def get_weighted_conformity(
    annotations: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """Computes conformity for each annotator. Works only for binary classification problems."""
    df = annotations.copy()

    mean_scores = df.groupby("text_id")[columns].mean()
    mean_scores = mean_scores.rename(columns=lambda col: col + "_mean")
    df = df.merge(mean_scores.reset_index())

    conformity_dfs = []
    for col in columns:
        df[f"{col}_annotator_agreement"] = df[f"{col}_mean"]
        df.loc[df[col] == 0, f"{col}_annotator_agreement"] = (
            1.0 - df[f"{col}_annotator_agreement"]
        )

        positive_df = df[df[col] == 1]
        negative_df = df[df[col] == 0]

        col_conformity_df = (
            df.groupby(["annotator_id"])[[f"{col}_annotator_agreement"]]
            .mean()
            .rename(columns=lambda _: col + "_w_conformity")
            .reset_index()
        )

        pos_col_conformity_df = (
            positive_df.groupby(["annotator_id"])[[f"{col}_annotator_agreement"]]
            .mean()
            .rename(columns=lambda _: col + "_pos_w_conformity")
            .reset_index()
        )
        neg_col_conformity_df = (
            negative_df.groupby(["annotator_id"])[[f"{col}_annotator_agreement"]]
            .mean()
            .rename(columns=lambda _: col + "_neg_w_conformity")
            .reset_index()
        )

        conformity_df = col_conformity_df.merge(pos_col_conformity_df, how="left")
        conformity_df = conformity_df.merge(neg_col_conformity_df, how="left")
        conformity_dfs.append(conformity_df.set_index("annotator_id"))

    return pd.concat(conformity_dfs, axis=1).fillna(1.0)
