import logging
import re

import pandas as pd

from sentify import FIGURE_DIR, DATA_PATH, EXPERIMENT_DIR
from sentify.cartography.data_map_plots import plot_data_map_with_hist, save_plot, plot_data_map

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
    'imdb': ([-0.008, 0.49], [-0.008, 1.01]),
    'sentiment140': ([-0.008, 0.49], [-0.008, 1.01]),
    'MHS': ([-0.008, 0.49], [-0.008, 1.01]),
    'MHS_hatespeech': ([-0.008, 0.49], [-0.008, 1.01]),
    'MHS_humiliate': ([-0.008, 0.49], [-0.008, 1.01]),
    'MHS_insult': ([-0.008, 0.49], [-0.008, 1.01]),
    'MHS_violence': ([-0.008, 0.49], [-0.008, 1.01]),
}

DATASET_NAME = 'MHS_hatespeech'
METHOD = 'hubi_medium'

EXP_NAME = f'{DATASET_NAME}_{METHOD}'
PERSEMO_DIR = DATA_PATH.joinpath('training_dynamics_persemo')
train_dy_file = sorted(
    list(PERSEMO_DIR.joinpath(DATASET_NAME).glob(f'{METHOD}_td_metrics*.jsonl')),
    key=lambda f: int(re.findall(r'\d+', str(f))[-1]),
)[-1]
# train_dy_file = sorted(
#     list(EXPERIMENT_DIR.joinpath(EXP_NAME).glob(f'td_metrics*.jsonl')),
#     key=lambda f: int(re.findall(r'\d+', str(f))[-1]),
# )[-1]

df_metrics = pd.read_json(
    train_dy_file,
    orient='records',
    lines=True,
)
df_metrics['model'] = MODEL_MAP[METHOD]

figure = plot_data_map_with_hist(
    df_metrics=df_metrics,
    model='',
    title=EXP_NAME,
)
filename = FIGURE_DIR.joinpath(f'datamap_{DATASET_MAP[DATASET_NAME]}_{MODEL_MAP[METHOD]}.pdf')
save_plot(filename=filename, fig=figure)

# compact version
xlim, ylim = LIMIT_MAP[DATASET_NAME]
figure = plot_data_map(
    df_metrics=df_metrics,
    add_legend=True,
    xlim=xlim,
    ylim=ylim,
    figsize=(7, 6),
    x_label='',
    y_label='',
)
filename = FIGURE_DIR.joinpath(
    f'datamap_compact_{DATASET_MAP[DATASET_NAME]}_{MODEL_MAP[METHOD]}.pdf'
)
save_plot(filename=filename, fig=figure)
