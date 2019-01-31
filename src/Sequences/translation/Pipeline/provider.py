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

class _Provider(object):
    def __init__(self, load_location, eval_proportion=0.2):
        self.load_location = load_location

        training_data, evaluation_data =


    def read_dataset(self, filenames, mode, config):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_example, num_parallel_calls=config['num_cpus'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # Train indefinitely
            dataset = dataset.shuffle(buffer_size=10 * config['batch_size'])
        else:
            num_epochs = 1  #
        dataset = dataset.repeat(num_epochs).batch(config['batch_size'])
        return dataset.make_one_shot_iterator().get_next()


    def _parse_example(self, example_proto):
        pass

    ################################################
    # Split dataset
    ################################################

    