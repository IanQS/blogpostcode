import tensorflow as tf
import logging

from Sequences.translation.Logger.logger import logging_setup
from Sequences.translation.Models.architectures import model_builder
from Sequences.translation.Pipeline.dataset import Dataset


class Orchestrator(object):
    def __init__(self, sess, model_config:dict, dataset_config:dict, logger_mode=logging.INFO):
        logging_setup(logger_mode)
        self.logger = logging.getLogger(__name__)
        self.sess = sess
        self.ds = Dataset(**dataset_config)
        self.model = model_builder(model_config)

    def build_dataset(self, raw_data_location, overwrite):
        self.ds.generate_records(raw_data_location, overwrite=overwrite)


    def train_and_evaluate(self, hparams: dict):
        """
        :param hparams: config not tied to a specific model e.g: batch_size, num_epochs vs. number of layers
        :return:
        """
        train_spec, eval_spec = self.ds.generate_specs(hparams)
        tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)



if __name__ == '__main__':

    # DEFAULTS
    LOAD_LOC = '../../datasets/translation/split/'
    SAVE_LOC_RECORDS = '../../datasets/translation/records/'
    SAVE_LOC_NPY = '../../datasets/translation/npz/'

    PRODUCER_CONSTRUCT_PATTERN = {
        'src': 'europarl-v7.sv-en.en{}',
        'target': 'europarl-v7.sv-en.sv{}'
    }

    DATASET_DEFAULTS = {
        'num_cpus': 8,
        'batch_size': 32
    }
    
    use_raw = True
    sess = tf.InteractiveSession()
    raw_data_location = '/'.join(LOAD_LOC.split('/')[1:])
    processed_data_location = '/'.join((SAVE_LOC_NPY if use_raw else SAVE_LOC_RECORDS).split('/')[1:])

    #############################################
    # Configs
    #   TODO: Load in via YAML
    #############################################

    MODEL_CONFIG = {
        'name' :'Transformer',
        'sess' :sess,

        # hparams specific to this model

        'embedding': {
            'top_k': 1000,
            'word_index': None,
            'embedding_path': None,
            'embedding_dim': None
        },

        'model_hparams':{
            'thing': 100
        }
    }

    DATASET_CONFIG = {
        'sess': sess,

        'producer_config' : {
            'glob_pattern': PRODUCER_CONSTRUCT_PATTERN,
            'generate_raw': use_raw,
            'save_location': processed_data_location,
            'write_condition': None,
            'to_process_location': None
        },

        'provider_config': {
            'use_raw': use_raw,
            'load_location': processed_data_location,
            'eval_proportion': 0.2,
            'padword': '<<<',
            'vocab_file_path': '.',
            'max_seq_len': 40
        }
    }

    # Hparams not specific to a model
    H_PARAMS = {
        'batch_size': 64,
        'num_epochs': 100,
        'eval_proportion': 0.2,
    }
    
    orchestrator = Orchestrator(sess, MODEL_CONFIG, DATASET_CONFIG)


    print(orchestrator.ds.generate_specs(H_PARAMS))
    #orchestrator.build_dataset(raw_data_location, overwrite=False)

    # Finally, train, evaluate, and deploy
    # orchestrator.train_and_evaluate()
