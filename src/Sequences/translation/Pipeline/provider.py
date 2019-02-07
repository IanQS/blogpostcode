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
        self.use_raw = use_raw

    def get_input_handlers(self):
        training_input_handlers = self._get_training_handlers()
        serving_input_handlers = self._get_serving_handlers()
        return training_input_handlers, serving_input_handlers


    def _get_training_handlers(self, source, target, batch_size, mode):
        x = tf.constant(source)

        # Map text to sequence of word-integers and pad
        x = self._vectorize_sentences(x)

        # Create tf.data.Dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((x, target))

        # Pad to constant length
        dataset = dataset.map(self._pad)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # loop indefinitley
            dataset = dataset.shuffle(buffer_size=50000)  # our input is already shuffled so this is redundant
        else:
            num_epochs = 1

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset

    def _get_serving_handlers(self):
        feature_placeholder = tf.placeholder(tf.string, [None])
        features = self._vectorize_sentences(feature_placeholder)
        return tf.estimator.export.TensorServingInputReceiver(features, feature_placeholder)

    def _vectorize_sentences(self, sentences):
        # 1. Remove punctuation
        sentences = tf.regex_replace(sentences, '[[:punct:]]', ' ')

        # 2. Split string tensor into component words
        words = tf.string_split(sentences)
        words = tf.sparse_tensor_to_dense(words, default_value=PADWORD)

        # 3. Map each word to respective integer
        table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=VOCAB_FILE_PATH,
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
        padded = tf.pad(without_zeros, [[MAX_SEQUENCE_LENGTH, 0]])  # pad out with zeros
        padded = padded[-MAX_SEQUENCE_LENGTH:]  # slice to constant length
        return (padded, label)

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