from tkinter import N
from typing import *
import argparse
import os

import pandas as pd
import numpy as np
from parso import parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sep', type=str, default='\t', help='Separator for reading csv/tsv etc')
    parser.add_argument('--comments_path', type=str, help='Path to commets file')
    parser.add_argument('--annotations_path', type=str, help='Path to annotations file')
    parser.add_argument('--annotators_data_path', type=str, help='Path to annotators data file')
    parser.add_argument('--annotation_column', type=str, help='Name of column with annotators id', default='worker_id')
    parser.add_argument('--comment_column', type=str, help='Name of column with text id', default='text_id')
    parser.add_argument('--n_annotations', type=int, help='Minimal number of annotations to keep user', default=40)
    parser.add_argument('--score_column', type=str, help='Name of label columns (score)')


def remove_users_with_less_than_n_annotations(
    df: pd.DataFrame,
    annotation_column: str, 
    comment_column: str, 
    n: int) -> pd.DataFrame:
    counted_annotations = df.groupby(annotation_column).count()
    users_to_discard = counted_annotations[counted_annotations[comment_column] < n].index
    processed_df = df[~df.worker_id.isin(users_to_discard)]
    return processed_df


def remove_users_who_voted_the_same(df: pd.DataFrame, annotation_column: str, score_column: str) -> pd.DataFrame:
    annotated_all_same = df.groupby(annotation_column).std()
    users_to_discard = annotated_all_same[annotated_all_same[score_column].round(4) == 0.0].index  
    processed_df = df[~df.worker_id.isin(users_to_discard)]
    return processed_df



def main() -> None:
    args = parse_args()
    comments_df = pd.read_csv(args.comments_path, sep=args.sep)
    annotations_df = pd.read_csv(args.annotations_path, sep=args.sep)
    annotators_data_df = pd.read_csv(args.annotators_data_path, sep=args.sep)

    new_annotations_data = remove_users_with_less_than_n_annotations(
        annotations_df,
        args.annotation_column,
        args.comment_column,
        args.n_annotations
    )
    new_annotations_data = remove_users_who_voted_the_same(
        new_annotations_data,
        args.annotation_column,
        args.score_column
    )
    new_annotators_data = annotators_data_df[annotators_data_df[args.annotation_column].isin(
        pd.unique(new_annotations_data[args.annotation_column])
    )]

    comments_df.to_csv(args.comments_path, sep='\t', index=False)
    new_annotations_data.to_csv(args.annotations_path, sep='\t', index=False)
    new_annotators_data.to_csv(args.annotators_data_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
