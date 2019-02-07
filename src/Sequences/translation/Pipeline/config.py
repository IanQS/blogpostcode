import logging
from pathlib import Path
import sys
import tensorflow as tf
import os
import logging.config
import yaml

LOAD_LOC = '../../datasets/translation/split/'
SAVE_LOC = '../../datasets/translation/records/'

pattern = {
    'src': 'europarl-v7.sv-en.en{}',
    'target': 'europarl-v7.sv-en.sv{}'
}

DATASET_DEFAULTS = {
    'num_cpus': 8,
    'batch_size': 32
}

def logging_setup(default_path='logging.yaml', default_level=logging.INFO,
                  env_key='LOG_CFG'):
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
