import logging
import re

import pandas as pd

from sentify import EXPERIMENT_DIR, FIGURE_DIR, DATA_PATH
from sentify.cartography.data_map_plots import plot_data_map_pair, save_plot, plot_change_data_map
from sentify.cartography.data_utils import get_difference_dataframe, get_difference_user_agg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    'baseline': 'Baseline',
    'user_identifier': 'UserIdentifier',
    'retriever': 'HumAnn',
    'hubi_medium': 'HuBi-Medium',
    'transformer_user_id': 'UserId',
}
DATASET_MAP = {
    'imdb': 'IMDB',
    'MHS': 'MHS',
    'sentiment140': 'Sentiment140',
    'MHS_insult': 'MHS_insult',
    'MHS_hatespeech': 'MHS_hatespeech',
    'MHS_violence': 'MHS_violence',
    'MHS_humiliate': 'MHS_humiliate',
}

LIMIT_MAP = {
    'imdb': ([-0.5, 0.2], [-1.0, 0.5]),
    'sentiment140': ([-0.5, 0.4], [-0.9, 0.85]),
    'MHS': ([-0.42, 0.5], [-0.95, 0.95]),
    'MHS_hatespeech': ([-0.42, 0.5], [-0.95, 0.95]),
    'MHS_humiliate': ([-0.42, 0.5], [-0.95, 0.95]),
    'MHS_insult': ([-0.42, 0.5], [-0.95, 0.95]),
    'MHS_violence': ([-0.42, 0.5], [-0.95, 0.95]),
}
LIMIT_USERS_MAP = {
    'imdb': ([-0.05, 0.1], [-0.2, 0.1]),
    # 'imdb': ([-0.3, 0.1], [-0.6, 0.1]),
    'sentiment140': ([-0.16, 0.045], [-0.45, 0.17]),
    'MHS': ([-0.25, 0.25], [-0.55, 0.5]),
    'MHS_hatespeech': ([-0.25, 0.25], [-0.55, 0.5]),
    'MHS_humiliate': ([-0.25, 0.25], [-0.55, 0.5]),
    'MHS_insult': ([-0.25, 0.25], [-0.55, 0.5]),
    'MHS_violence': ([-0.25, 0.25], [-0.55, 0.5]),
}

mhs_dimension = 'hatespeech'
DATASET_NAME = f'MHS_{mhs_dimension}'
# DATASET_NAME = 'MHS'
MODEL_1 = 'baseline'
MODEL_2 = 'hubi_medium'
EXP1_NAME = f'{DATASET_NAME}_{MODEL_1}'
# EXP2_NAME = f'{DATASET_NAME}_{MODEL_2}'
EXP2_NAME = f'{DATASET_NAME}'
train_dy_file1 = sorted(
    list(EXPERIMENT_DIR.joinpath(EXP1_NAME).glob('td_metrics*.jsonl')),
    key=lambda f: int(re.findall(r'\d+', str(f))[-1]),
)[-1]

# train_dy_file2 = sorted(
#     list(EXPERIMENT_DIR.joinpath(EXP2_NAME).glob(f'td_metrics*.jsonl')),
#     key=lambda f: int(re.findall(r'\d+', str(f))[-1]),
# )[-1]

PERSEMO_DIR = DATA_PATH.joinpath('training_dynamics_persemo')
train_dy_file2 = sorted(
    list(PERSEMO_DIR.joinpath(DATASET_NAME).glob(f'{MODEL_2}_td_metrics*.jsonl')),
    key=lambda f: int(re.findall(r'\d+', str(f))[-1]),
)[-1]

df1_metrics = pd.read_json(
    train_dy_file1,
    orient='records',
    lines=True,
)
df1_metrics['model'] = MODEL_MAP[MODEL_1]

df2_metrics = pd.read_json(
    train_dy_file2,
    orient='records',
    lines=True,
)
df2_metrics['model'] = MODEL_MAP[MODEL_2]

# SAMPLES: calculate changes in values
df_diff, quarter_counts = get_difference_dataframe(df1=df1_metrics, df2=df2_metrics)
xlim, ylim = LIMIT_MAP[DATASET_NAME]

fig = plot_change_data_map(
    df_diff,
    xlim=xlim,
    ylim=ylim,
    quarter_counts=quarter_counts,
    x_label='',
    y_label='',
    # hue_label='',
)
filename = FIGURE_DIR.joinpath(f'diff_{DATASET_NAME}_{MODEL_MAP[MODEL_1]}_{MODEL_MAP[MODEL_2]}.pdf')
save_plot(fig, filename=filename)

# USER: aggregated differential data cartography
df_user_agg, quarter_user_counts = get_difference_user_agg(
    data_name="MHS",
    df_diff=df_diff,
    label=mhs_dimension,
)
xlim, ylim = LIMIT_USERS_MAP[DATASET_NAME]

fig = plot_change_data_map(
    df_user_agg,
    xlim=xlim,
    ylim=ylim,
    hue_dim='entropy',
    x_dim='var_diff_mean',
    y_dim='conf_diff_mean',
    quarter_counts=quarter_user_counts,
    # x_label='',
    # y_label='',
    hue_label='',
)
filename = FIGURE_DIR.joinpath(
    f'diff_users_{DATASET_NAME}_{MODEL_MAP[MODEL_1]}_{MODEL_MAP[MODEL_2]}.pdf'
)
save_plot(fig, filename=filename)
