"""
Do not use this file directly. Instead, interact with it via the dataset class

Uses tweaked code from https://github.com/IanQS/blogpostcode/tree/master/src/Tf_Exploration/exploration

Author: Ian Q.

Notes:
    Purpose:
        - generates the tf data Datasets

        - always stay on to accumulate new sentences that it sees, and write to disk
        or some location

        - reads in data as a list, and writes them based on the size

"""


import tensorflow as tf
import tqdm
import os

def bytes_feature(value: str):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class _Provider(object):
    def __init__(self, condition, train_loc, test_loc):
        self.write_condition = condition
        self.train_loc = train_loc
        self.test_loc = test_loc

    ################################################
    #Public Interace
    ################################################

    def generate_records(self, prefixes: dict = None):

        # Is serving and we want to hook into saving
        if prefixes is None:
            if not self.write_condition(self):
                return

        # Else, continue as normal


    ################################################
    #Private Interace
    ################################################

    def _dataset_creation(self, data):
        idx = 0
        collection = {}
        for prototype in self.features:
            datum = [data[idx]] if prototype.shape == 1 else data[idx:idx + prototype.shape]
            encoded_feature = bytes_feature(value=datum)
            collection[prototype.name] = encoded_feature
            idx += prototype.shape
        return collection

    def _write(self, filename, indices, feature_proto, loaded):
        # Round to the previous hour
        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in tqdm.tqdm(indices):
                datum = loaded[i, :]
                feature = feature_proto.dataset_creation(datum)
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())


    ################################################
    #Create location
    ################################################