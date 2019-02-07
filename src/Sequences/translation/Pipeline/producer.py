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

    Generic:
        - if use_raw in dataset is True, this object is not instantiated. Instead, we
        read directly from
"""
import numpy as np

import tensorflow as tf
import tqdm
import os
import errno
from typing import Callable
from natsort import natsorted
import logging


def bytes_feature(value: list):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class _Producer(object):
    def __init__(self, save_location: str, write_condition: Callable = None,
                 to_process_location: str=None, generate_raw=False):
        """
        Create the Provider object

        :param write_condition:
            condition for when to generate records.
            only triggered when we don't explicitly receive data
        """
        self.logger = logging.getLogger(__name__)
        self.save_loc = save_location

        self.write_condition = write_condition
        self.storage = None if self.write_condition is None else []
        if write_condition is not None:
            err_m = 'Write condition was specified, but location to save to: "to_process_location" was not specified'
            self.logger.critical(err_m)
            self.to_process_location = to_process_location

        self._raw = generate_raw
        self.__create_save_location()

    ################################################
    # Public Interace
    ################################################

    def generate_records(self, load_from: str = None, pattern=None, overwrite=True) -> None:
        # Is serving and we want to hook into saving
        if load_from is None:
            self.logger.info('Data Source is None. Writing data received from serving')
            if self.write_condition(self):
                self.__write_txt(self.storage)
            return

        # Else, continue as normal
        filenames = self.__glob_records(load_from, pattern)
        self.__generate_records(filenames=filenames, overwrite=overwrite)

    ################################################
    # Private Interace
    ################################################

    def __generate_records(self, filenames: list, overwrite: bool) -> None:
        curr_index = 0  # we split the dataset and started from 1
        if not overwrite:
            _ = self.__get_max_index(self.save_loc)
            if _ != 0:
                if self._raw:
                    curr_index = int(_.split('.npy')[0])
                else:
                    curr_index = int(_.split('.tfrecord')[0])
            else:
                curr_index = 0
            filenames = filenames[curr_index:]  # -1 because the datasets are stored 1 index

        self.logger.info('Generating raw' if self._raw else 'Generating tfRecord')
        for src_target in tqdm.tqdm(filenames):
            self.__read_and_write(src_target, curr_index)
            curr_index += 1

    def __read_and_write(self, src_target, curr_index):

        # Read data first
        accum_source, accum_target = [], []

        for mode, storage in zip(src_target, [accum_source, accum_target]):
            with open(mode, 'rb') as f:
                for line in f.readlines():
                    storage.append(line.strip())

        # Pass list over to __write
        if self._raw:
            self.__write_raw(accum_source, accum_target, curr_index)
        else:
            self.__write_record(accum_source, accum_target, curr_index)


    def __write_raw(self, source:list, target: list, index):
        filename = '{}/{}.npy'.format(self.save_loc, index)
        dict_accum = {
            'source': source,
            'target': target
        }
        np.save(filename, dict_accum)

    def __write_record(self, source: list, target: list, index):
        filename = '{}/{}.tfrecord'.format(self.save_loc, index)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for ind_src, ind_targ in tqdm.tqdm(zip(source, target), desc='Writer'):
                features = {
                    'source': bytes_feature([ind_src]),
                    'target': bytes_feature([ind_targ])
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example_proto.SerializeToString())

    ################################################
    # Non-TF related code
    ################################################

    def __write_txt(self, data):
        _ = self.__get_max_index(self.to_process_location)
        if _ != 0:
            max_name = int(_.split('.txt')) + 1
        else:
            max_name = '0.txt'
        string_max_name = '{}.txt'.format(max_name)
        with open(os.path.join(self.to_process_location, string_max_name), 'w') as f:
            f.write(data)

    def __glob_records(self, location, pattern) -> list:
        i = 0
        accum = []
        while True:
            formatted_string = '0{}'.format(i) if i < 10 else i
            src_file = os.path.join(location, pattern['src'].format(formatted_string))
            targ_file = os.path.join(location, pattern['target'].format(formatted_string))

            if not os.path.exists(src_file) or not os.path.exists(targ_file):
                break

            accum.append([src_file, targ_file])
            i += 1

        return accum

    def __create_save_location(self) -> None:
        try:
            os.mkdir(self.save_loc)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    def __get_max_index(self, path):
        contents = natsorted(os.listdir(path), reverse=True)
        try:
            max_name = contents[0]  # 1.npy/1.tensorflow
        except Exception as e:
            self.logger.info('Failed to get max index. Contents: {}'.format(contents))
            max_name = 0
        return max_name