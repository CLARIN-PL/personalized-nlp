from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
STORAGE_DIR = PROJECT_DIR / 'storage'
LOGS_DIR = STORAGE_DIR / 'logs'
CHECKPOINTS_DIR = STORAGE_DIR / 'checkpoints'

# path for all data ex. Wikidata, Unhealthy Conversations data etc
DATA_DIR = STORAGE_DIR / 'data'

# path for all outputs ex. Cartography outputs, save predictions outputs etc.
OUTPUTS_DIR_NAME =  STORAGE_DIR / 'outputs'
CARTOGRAPHY_OUTPUTS_DIR_NAME = OUTPUTS_DIR_NAME / 'cartography_outputs'
CARTOGRAPHY_TRAIN_DYNAMICS_DIR_NAME = 'train_dynamics'
CARTOGRAPHY_PLOTS_DIR_NAME = 'plots'
CARTOGRAPHY_METRICS_DIR_NAME = 'metrics'
CARTOGRAPHY_FILTER_DIR_NAME = 'filtered'

SAVE_PREDICTIONS_DIR_NAME = OUTPUTS_DIR_NAME / 'predictions'

EMOTIONS_DATA = STORAGE_DIR / 'emotions_data'
EMOTIONS_SIMPLE_DATA = STORAGE_DIR / 'emotions_simple_data'

AGGRESSION_URL = 'https://ndownloader.figshare.com/articles/4267550/versions/5'
ATTACK_URL = 'https://ndownloader.figshare.com/articles/4054689/versions/6'
TOXICITY_URL = 'https://ndownloader.figshare.com/articles/4563973/versions/2'

CBOW_EMBEDDINGS_PATH = STORAGE_DIR / 'word2vec' / 'kgr10.plain.cbow.dim300.neg10.bin'
SKIPGRAM_EMBEDDINGS_PATH = STORAGE_DIR / 'word2vec' / 'kgr10.plain.skipgram.dim300.neg10.bin'

EMBEDDINGS_SIZES = {
    'xlmr': 768,
    'bert': 768,
    'labse': 768, 
    'mpnet': 768,
    'random': 768,
    'skipgram': 300,
    'cbow': 300,
    'deberta': 1024,
    'roberta': 768
}

TRANSFORMER_MODEL_STRINGS = {
    'xlmr': 'xlm-roberta-base',
    'bert': 'bert-base-cased',
    'deberta': 'microsoft/deberta-large',
    'labse': 'sentence-transformers/LaBSE',
    'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'roberta': 'roberta-base'
}
