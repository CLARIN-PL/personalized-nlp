import pathlib
import argparse
import logging

import pandas as pd

from scripts.process_data.assign_folds import assign_folds
from scripts.process_data.reset_indexes import (
    reindex_texts_and_annotations,
    deduplicate_texts,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments for script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations_df_path",
        "-ap",
        type=str,
        dest="annotations_df_path",
        required=True,
        help="Path to annotations csv/tsv.",
    )
    parser.add_argument(
        "--texts_df_path",
        "-tp",
        type=str,
        dest="texts_df_path",
        required=False,
        default="",
        help="Path to texts csv/tsv. If not specified, enter single-file scenario.",
    )
    parser.add_argument(
        "--text_id_col",
        "-tic",
        type=str,
        required=True,
        help="Name of text idx column.",
    )
    parser.add_argument(
        "--text_col",
        "-tc",
        type=str,
        required=False,
        default="text",
        help="Name of text content column.",
    )
    parser.add_argument(
        "--annotator_col",
        "-ac",
        type=str,
        required=True,
        help="Name of annotator idx column.",
    )
    parser.add_argument(
        "--num_folds", "-nf", type=int, default=10, help="Number of folds to create."
    )
    return parser.parse_args()


def create_path(old_path: str, extra: str = "") -> pathlib.PosixPath:
    """Creates a path for new file
    Args:
        old_path (str): Old file's path.
        extra (str, optional): Extra word to add. Defaults to ''.
    Returns:
        str: New file's path.
    """
    extra = extra if extra == "" else f"_{extra}"
    base_filename = f"{pathlib.Path(old_path).stem}{extra}.csv"
    new_path = pathlib.Path(old_path).parents[0] / (base_filename)
    return new_path


def main():
    args = parse_args()
    annotations_df = pd.read_csv(
        args.annotations_df_path,
        sep="\t" if args.annotations_df_path.endswith(".tsv") else ",",
    )
    logger.info(f"Loaded annotations csv with {len(annotations_df)} annotations.")
    texts_df = pd.read_csv(
        args.texts_df_path,
        sep="\t" if args.annotations_df_path.endswith(".tsv") else ",",
    )
    logger.info(f"Loaded texts csv with {len(texts_df)} texts.")

    deduplicate_texts(texts_df, annotations_df, args.text_col, args.text_id_col)

    texts_df = texts_df.drop_duplicates(subset=[args.text_id_col])
    texts_df = texts_df.reset_index(drop=True)

    annotations_df_reindex, texts_df_reindex = reindex_texts_and_annotations(
        annotations_df=annotations_df,
        texts_df=texts_df,
        text_id_col=args.text_id_col,
        annotator_col=args.annotator_col,
    )
    logger.info(f"Reindexed annotations and texts.")

    annotations_text_folds = assign_folds(
        annotations_df=annotations_df_reindex.copy(),
        stratify_by=args.text_id_col,
        num_folds=args.num_folds,
    )
    logger.info(f"Created text folds.")

    annotations_user_folds = assign_folds(
        annotations_df=annotations_df_reindex,
        stratify_by=args.annotator_col,
        num_folds=args.num_folds,
    )
    logger.info(f"Created user folds.")

    text_folds_path = create_path(args.annotations_df_path, extra="texts_folds")
    annotations_text_folds.to_csv(text_folds_path, index=False)
    logger.info(f"Saved text folds to {text_folds_path}.")

    user_folds_path = create_path(args.annotations_df_path, extra="users_folds")
    annotations_user_folds.to_csv(user_folds_path, index=False)
    logger.info(f"Saved user folds to {user_folds_path}.")

    texts_reindex_path = create_path(args.texts_df_path, extra="processed")
    texts_df_reindex.to_csv(texts_reindex_path, index=False)
    logger.info(f"Saved reindexed texts to {texts_reindex_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
