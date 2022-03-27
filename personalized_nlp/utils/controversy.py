from typing import List
import pandas as pd

import numpy as np
from scipy.stats import entropy


def _entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def get_text_controversy(annotations: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return annotations.groupby("text_id")[score_column].apply(_entropy)


def get_conformity(annotations: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Computes conformity for each annotator. Works only for binary classification problems."""
    df = annotations.copy()

    mean_score = df.groupby("text_id").agg(score_mean=(columns, "mean"))
    df = df.merge(mean_score.reset_index())

    df["text_major_vote"] = (df["score_mean"] > 0.5).astype(int)

    df["is_major_vote"] = df["text_major_vote"] == df[columns]
    df["is_major_vote"] = df["is_major_vote"].astype(int)

    positive_df = df[df.text_major_vote == 1]
    negative_df = df[df.text_major_vote == 0]

    conformity_df = df.groupby("annotator_id").agg(conformity=("is_major_vote", "mean"))
    conformity_df["pos_conformity"] = positive_df.groupby("annotator_id").agg(
        pos_conformity=("is_major_vote", "mean")
    )
    conformity_df["neg_conformity"] = negative_df.groupby("annotator_id").agg(
        neg_conformity=("is_major_vote", "mean")
    )

    return conformity_df
