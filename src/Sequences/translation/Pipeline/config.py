import logging
from pathlib import Path
import sys
import tensorflow as tf

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

def logging_setup(level=logging.INFO):
    # Setup logging
    Path('results').mkdir(exist_ok=True)
    tf.logging.set_verbosity(level)
    handlers = [
        logging.FileHandler('results/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers