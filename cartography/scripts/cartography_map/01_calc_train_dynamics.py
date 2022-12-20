import logging

from sentify import DATA_PATH, EXPERIMENT_DIR
from sentify.cartography.metrics import compute_train_dy_metrics
from sentify.cartography.read_utils import read_training_dynamics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = 'MHS_violence'
METHOD = 'transformer_user_id'

EXP_NAME = f'{DATASET_NAME}_{METHOD}'
PERSEMO_DIR = DATA_PATH.joinpath('training_dynamics_persemo')
# TRAINING_DYNAMICS_DIR = EXPERIMENT_DIR.joinpath(EXP_NAME, 'training_dynamics')
TRAINING_DYNAMICS_DIR = PERSEMO_DIR.joinpath(DATASET_NAME, METHOD)

# maximum epochs for cartography maps
epoch_burn_out = None

train_dynamics_dict = read_training_dynamics(
    train_dynamics_dir=TRAINING_DYNAMICS_DIR,
    epoch_burn_out=epoch_burn_out,
)

df_metrics, df_train = compute_train_dy_metrics(
    training_dynamics=train_dynamics_dict,
    variability_include_ci=True,
    gold_label='gold',
)
num_epochs = df_metrics['epochs'][0]
# train_dy_filename = EXPERIMENT_DIR.joinpath(EXP_NAME, f"td_metrics{num_epochs}.jsonl")
train_dy_filename = TRAINING_DYNAMICS_DIR.parent.joinpath(f'{METHOD}_td_metrics{num_epochs}.jsonl')
df_metrics.to_json(
    train_dy_filename,
    orient='records',
    lines=True,
)
logger.info(f"Metrics based on Training Dynamics written to {train_dy_filename}")
