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
    'MHS': ([-0.42, 0.5], [-0.7, 0.9]),
    'MHS_hatespeech': ([-0.42, 0.5], [-0.7, 0.9]),
    'MHS_humiliate': ([-0.42, 0.5], [-0.7, 0.9]),
    'MHS_insult': ([-0.42, 0.5], [-0.7, 0.9]),
    'MHS_violence': ([-0.42, 0.5], [-0.7, 0.9]),
}
LIMIT_USERS_MAP = {
    'imdb': ([-0.05, 0.1], [-0.2, 0.1]),
    # 'imdb': ([-0.3, 0.1], [-0.6, 0.1]),
    'sentiment140': ([-0.16, 0.045], [-0.25, 0.17]),
    'MHS': ([-0.1, 0.17], [-0.1, 0.45]),
    'MHS_hatespeech': ([-0.1, 0.17], [-0.1, 0.45]),
    'MHS_humiliate': ([-0.1, 0.25], [-0.1, 0.45]),
    'MHS_insult': ([-0.1, 0.25], [-0.1, 0.45]),
    'MHS_violence': ([-0.1, 0.25], [-0.1, 0.45]),
}

mhs_dimension = 'hatespeech'
DATASET_NAME = f'MHS_{mhs_dimension}'
MODEL_2 = 'hubi_medium'
EXP2_NAME = f'{DATASET_NAME}'
PERSEMO_DIR = DATA_PATH.joinpath('training_dynamics_persemo')
train_dy_file2 = sorted(
    list(PERSEMO_DIR.joinpath(DATASET_NAME).glob(f'{MODEL_2}_td_metrics*.jsonl')),
    key=lambda f: int(re.findall(r'\d+', str(f))[-1]),
)[-1]


df2_metrics = pd.read_json(
    train_dy_file2,
    orient='records',
    lines=True,
)
# df2_metrics['model'] = MODEL_MAP[MODEL_2]
df2_metrics = df2_metrics.sort_values(by='guid')

df2_metrics['guid'] = df2_metrics['guid'].map(lambda x: int(f'1{x - df2_metrics.guid.min()}'))
df2_metrics.to_json(train_dy_file2, orient='records', lines=True)
