from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = PROJECT_DIR / 'storage'
LOGS_DIR = STORAGE_DIR / 'logs'
CHECKPOINTS_DIR = STORAGE_DIR / 'checkpoints'

EMOTIONS_DATA = STORAGE_DIR / 'emotions_data'
EMOTIONS_SIMPLE_DATA = STORAGE_DIR / 'emotions_simple_data'

AGGRESSION_URL = 'https://ndownloader.figshare.com/articles/4267550/versions/5'
ATTACK_URL = 'https://ndownloader.figshare.com/articles/4054689/versions/6'
TOXICITY_URL = 'https://ndownloader.figshare.com/articles/4563973/versions/2'

CBOW_EMBEDDINGS_PATH = STORAGE_DIR / 'word2vec' / 'kgr10.plain.cbow.dim300.neg10.bin'
SKIPGRAM_EMBEDDINGS_PATH = STORAGE_DIR / 'word2vec' / 'kgr10.plain.skipgram.dim300.neg10.bin'