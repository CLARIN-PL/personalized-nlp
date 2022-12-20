import logging
import os
import shutil
from typing import Dict, Optional

import pandas as pd
from scipy.stats import entropy

from sentify import DATASETS_PATH

logger = logging.getLogger(__name__)


def _normalize_by_epochs(
        df: pd.DataFrame,
        num_epochs: int,
        metric: str = 'correctness',
) -> pd.DataFrame:
    # Normalize correctness to a value between 0 and 1
    # (1 indicates always predicted correctly label across epochs)
    df = df.assign(metric_frac=lambda d: d[metric] / num_epochs)
    df[metric] = [float(f'{x:.1f}') for x in df['metric_frac']]
    return df


def get_difference_dataframe(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df1 = df1.sort_values(by='guid')
    df1 = _normalize_by_epochs(
        df=df1,
        num_epochs=df1['epochs'][0],
        metric='correctness',
    )
    df2 = df2.sort_values(by='guid')
    df2 = _normalize_by_epochs(
        df=df2,
        num_epochs=df2['epochs'][0],
        metric='correctness',
    )

    df_merged = df1.merge(right=df2, on='guid', suffixes=('_1', '_2'))
    df_merged['conf_diff'] = df_merged['confidence_2'] - df_merged['confidence_1']
    df_merged['var_diff'] = df_merged['variability_2'] - df_merged['variability_1']
    df_merged['corr_diff'] = df_merged['correctness_2'] - df_merged['correctness_1']

    quarter_counts = {
        'quarter_I': len(df_merged[(df_merged['conf_diff'] > 0) & (df_merged['var_diff'] > 0)]),
        'quarter_II': len(df_merged[(df_merged['conf_diff'] < 0) & (df_merged['var_diff'] > 0)]),
        'quarter_III': len(df_merged[(df_merged['conf_diff'] < 0) & (df_merged['var_diff'] < 0)]),
        'quarter_IV': len(df_merged[(df_merged['conf_diff'] > 0) & (df_merged['var_diff'] < 0)]),
    }

    return df_merged, quarter_counts


def get_difference_user_agg(
        data_name: str,
        df_diff: pd.DataFrame,
        label: Optional[str] = 'label',
) -> tuple[pd.DataFrame, dict]:
    df_train = pd.read_csv(
        DATASETS_PATH.joinpath(data_name, 'train.tsv'),
        sep='\t',
        engine='python',
    )
    df_merged = df_train.merge(df_diff, left_on='index', right_on='guid', how='inner')

    df_group = df_merged.groupby(by=['username'], as_index=False).agg(
        {
            'conf_diff': ['mean', 'std'],
            'var_diff': ['mean', 'std'],
            'corr_diff': ['mean', 'std'],
        }
    )
    df_group.columns = ['_'.join(col) for col in df_group.columns.values]

    df_entropy = (
        df_merged.groupby('username')[label]
        .apply(lambda x: entropy(x.value_counts()))
        .reset_index()
    )
    df_entropy = df_entropy.rename(columns={label: 'entropy'})

    min_entropy = df_entropy[['entropy']].min()
    max_entropy = df_entropy[['entropy']].max()
    df_entropy[['entropy']] = (df_entropy[['entropy']] - min_entropy) / (max_entropy - min_entropy)

    df_user_agg = df_group.merge(df_entropy, left_on='username_', right_on='username')

    quarter_counts = {
        'quarter_I': len(
            df_user_agg[(df_user_agg['conf_diff_mean'] > 0) & (df_user_agg['var_diff_mean'] > 0)]
        ),
        'quarter_II': len(
            df_user_agg[(df_user_agg['conf_diff_mean'] < 0) & (df_user_agg['var_diff_mean'] > 0)]
        ),
        'quarter_III': len(
            df_user_agg[(df_user_agg['conf_diff_mean'] < 0) & (df_user_agg['var_diff_mean'] < 0)]
        ),
        'quarter_IV': len(
            df_user_agg[(df_user_agg['conf_diff_mean'] > 0) & (df_user_agg['var_diff_mean'] < 0)]
        ),
    }
    return df_user_agg, quarter_counts


def convert_tsv_entries_to_dataframe(tsv_dict: Dict, header: str) -> pd.DataFrame:
    """
    Converts entries from TSV file to Pandas DataFrame for faster processing.
    """
    header_fields = header.strip().split("\t")
    data = {header: [] for header in header_fields}

    for line in tsv_dict.values():
        fields = line.strip().split("\t")
        assert len(header_fields) == len(fields)
        for field, header in zip(fields, header_fields):
            data[header].append(field)

    df = pd.DataFrame(data, columns=header_fields)
    return df


def copy_dev_test(task_name: str, from_dir: os.path, to_dir: os.path, extension: str = ".tsv"):
    """
    Copies development and test sets (for data selection experiments) from `from_dir` to `to_dir`.
    """
    if task_name == "MNLI":
        dev_filename = "dev_matched.tsv"
        test_filename = "dev_mismatched.tsv"
    elif task_name in ["SNLI", "QNLI", "WINOGRANDE"]:
        dev_filename = f"dev{extension}"
        test_filename = f"test{extension}"
    else:
        raise NotImplementedError(f"Logic for {task_name} not implemented.")

    dev_path = os.path.join(from_dir, dev_filename)
    if os.path.exists(dev_path):
        shutil.copyfile(dev_path, os.path.join(to_dir, dev_filename))
    else:
        raise ValueError(f"No file found at {dev_path}")

    test_path = os.path.join(from_dir, test_filename)
    if os.path.exists(test_path):
        shutil.copyfile(test_path, os.path.join(to_dir, test_filename))
    else:
        raise ValueError(f"No file found at {test_path}")


def read_jsonl(file_path: str, key: str = "pairID"):
    """
    Reads JSONL file to recover mapping between one particular key field
    in the line and the result of the line as a JSON dict.
    If no key is provided, return a list of JSON dicts.
    """
    df = pd.read_json(file_path, lines=True)
    records = df.to_dict('records')
    logger.info(f"Read {len(records)} JSON records from {file_path}.")

    if key:
        assert key in df.columns
        return {record[key]: record for record in records}
    return records
