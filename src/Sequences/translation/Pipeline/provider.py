"""
Do not use this file directly. Instead, interact with it via the dataset class

Uses tweaked code from https://github.com/IanQS/blogpostcode/tree/master/src/Tf_Exploration/exploration

Author: Ian Q.

Notes:
    Purpose:
        - Provides boilerplate to consume the generated datasets

        - provide whatever is required for inference time and serving
"""
import time

import tensorflow as tf
import os
import logging
import numpy as np

class _Provider(object):
    def __init__(self, load_location, eval_proportion=0.2, use_raw=False, padword=None, vocab_file_path=None, max_seq_len=None):
        self.logger = logging.getLogger(__name__)
        if padword is None or vocab_file_path is None or max_seq_len is None:
            self.logger.critical('padword or vocab_file_path or max_seq_len was none. All should be filled')
            raise Exception('Provider Crash')
        self.load_location = load_location
        self.eval_proportion = eval_proportion
        self.use_raw = use_raw
        self.padword = padword
        self.vocab_file_path = vocab_file_path
        self.max_seq_len = max_seq_len

    def generate_specs(self, hparams: dict):
        """Generate the eval and train estimator.Specs
        :return:
        """
        proportion = hparams.get('eval_proportion', self.eval_proportion)
        train_source, train_target, test_source, test_target = self._get_data(proportion)

        train_steps = hparams['num_epochs'] * len(train_source) / hparams['batch_size']
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda:self._get_handlers(
                train_source,
                train_target,
                hparams['batch_size'],
                mode=tf.estimator.ModeKeys.TRAIN),
            max_steps=train_steps
        )

        # Create EvalSpec
        exporter = tf.estimator.LatestExporter('exporter', self._get_serving_handlers)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda:self._get_handlers(
                test_source,
                test_target,
                hparams['batch_size'],
                mode=tf.estimator.ModeKeys.EVAL),
            steps=None,
            exporters=exporter,
            start_delay_secs=10,
            throttle_secs=10
        )

        return train_spec, eval_spec

    #############################################
    # tf Data Handlers
    #############################################
    
    def _get_handlers(self, source, target, batch_size, mode):
        x = tf.constant(source)

        # Map text to sequence of word-integers and pad
        x = self._vectorize_sentences(x)

        # Create tf.data.Dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((x, target))

        # Pad to constant length
        dataset = dataset.map(self._pad)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=batch_size * 100)
        else:
            num_epochs = 1

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset

    def _get_serving_handlers(self):
        feature_placeholder = tf.placeholder(tf.string, [None])
        features = self._vectorize_sentences(feature_placeholder)
        return tf.estimator.export.TensorServingInputReceiver(features, feature_placeholder)


    #############################################
    # tf Pre-processing utilities
    #############################################

    def _vectorize_sentences(self, sentences):
        # 1. Remove punctuation
        sentences = tf.regex_replace(sentences, '[[:punct:]]', ' ')

        # 2. Split string tensor into component words
        words = tf.string_split(sentences)
        words = tf.sparse_tensor_to_dense(words, default_value=self.padword)

        # 3. Map each word to respective integer
        table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self.vocab_file_path,
            num_oov_buckets=0,
            vocab_size=None,
            default_value=0,  # for words not in vocabulary (OOV)
            key_column_index=0,
            value_column_index=1,
            delimiter=',')
        numbers = table.lookup(words)

        return numbers

    def _pad(self, feature, label):
        # 1. Remove 0s which represent out of vocabulary words
        nonzero_indices = tf.where(tf.not_equal(feature, tf.zeros_like(feature)))
        without_zeros = tf.gather(feature, nonzero_indices)
        without_zeros = tf.squeeze(without_zeros, axis=1)

        # 2. Prepend 0s till MAX_SEQUENCE_LENGTH
        padded = tf.pad(without_zeros, [[self.max_seq_len, 0]])  # pad out with zeros
        padded = padded[-self.max_seq_len:]  # slice to constant length
        return (padded, label)

    #############################################
    # non-tf Data Handlers
    #############################################

    def _get_data(self, eval_proportion):
        train_source = []
        train_target = []

        test_source = []
        test_target = []

        train_files, test_files = self.__split_train_eval(eval_proportion)
        self.logger.debug('Train files: {}'.format(train_files))
        self.logger.debug('Test files: {}'.format(test_files))

        for f_name in train_files:
            data = np.load(f_name)
            train_source.extend(data['source'])
            train_target.extend(data['target'])
            data.close()

        for f_name in test_files:
            data = np.load(f_name)
            test_source.extend(data['source'])
            test_target.extend(data['target'])
            data.close()

        return train_source, train_target, test_source, test_target

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