"""
Do not use this file directly. Instead, interact with it via the dataset class

Uses tweaked code from https://github.com/IanQS/blogpostcode/tree/master/src/Tf_Exploration/exploration

Author: Ian Q.

Notes:
    Purpose:
        - Provides boilerplate to consume the generated datasets

        - provide whatever is required for inference time and serving
"""

import tensorflow as tf
import os
import logging

class _Provider(object):
    def __init__(self, load_location, eval_proportion=0.2, use_raw=False):
        self.load_location = load_location
        self.eval_proportion = eval_proportion
        self.use_raw = use_raw
        self.logger = logging.getLogger(__name__)


    def construct_datasets(self, config):
        training_data_files, evaluation_data_files = self._get_split()

        train_dataset = self._read_dataset(training_data_files, tf.estimator.ModeKeys.TRAIN, config)

        eval_dataset = self._read_dataset(evaluation_data_files, tf.estimator.ModeKeys.EVAL, config)

        self.logger.info('Done reading!')
        return {'train_files': training_data_files, 'train_dataset': train_dataset,
                'eval_files': evaluation_data_files, 'eval_dataset': eval_dataset
                }


    def _get_split(self):
        training_data, evaluation_data = self.__split_train_eval(self.eval_proportion)
        return training_data, evaluation_data

    def _read_dataset(self, filenames, mode, config):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.__parse_example, num_parallel_calls=config['num_cpus'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # Train indefinitely
            dataset = dataset.shuffle(buffer_size=10 * config['batch_size'])
        else:
            num_epochs = 1  #
        dataset = dataset.repeat(num_epochs).batch(config['batch_size'])
        return dataset.make_one_shot_iterator().get_next()


    def __parse_example(self, example_proto):
        features = {
            'source': tf.VarLenFeature(dtype=tf.string),
            'target': tf.VarLenFeature(dtype=tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, feature)
        
#         sentences = tf.regex_replace(parsed_features['source'], '[[:punct:]]', ' ')
#         words = tf.string_split(sentences)
#         words = tf.sparse_tensor_to_dense(words, default_value=)
        
        
        return parsed_features['source'], parsed_features['target']

    ################################################
    # Split dataset
    ################################################

    def __split_train_eval(self, proportion):
        test_files = []
        train_files = []
        for f_name in os.listdir(self.load_location):
            hashed_name = (hash(f_name) % 10) / 10

            fixed_path = os.path.join(self.load_location, f_name)
            if hashed_name < proportion:
                test_files.append(fixed_path)
            else:
                train_files.append(fixed_path)

        return train_files, test_files