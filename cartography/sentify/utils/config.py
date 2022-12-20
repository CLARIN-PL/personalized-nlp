import yaml

from sentify import PARAMS_FILE


def load_config():
    with PARAMS_FILE.open('r') as f:
        config = yaml.safe_load(f)
        return config
