import tensorflow as tf
import logging

from Sequences.translation.Logger.logger import logging_setup
from Sequences.translation.Models.architectures import model_builder
from Sequences.translation.Pipeline.dataset import Dataset


class Orchestrator(object):
    def __init__(self, sess, model_config:dict, dataset_config:dict):
        logging_setup()
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
    from Sequences.translation.Pipeline.config import LOAD_LOC, SAVE_LOC_RECORDS, \
        PRODUCER_CONSTRUCT_PATTERN, SAVE_LOC_NPY
    
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


        'model_hparams' :{

        },

        'hparams':{
            'batch_size': 64,
            'num_epochs': 100
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
            'vocab_file_path': None,
            'max_seq_len': 40
        }
    }
    
    orchestrator = Orchestrator(sess, MODEL_CONFIG, DATASET_CONFIG)

    # orchestrator.build_dataset(raw_data_location, overwrite=False)

    # Finally, train, evaluate, and deploy
    # orchestrator.train_and_evaluate()
