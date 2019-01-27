"""
model_utils.py
    - for use with tf_exploration. Contains classes from Notebook1-data_exploration (for use in Notebook2)
    
    - lets us focus on notebook 2. Will serve as jumping point for the production series
"""

# Dataset creation
# Dataset reading
# placeholder definition

import tensorflow as tf
from typing import Tuple
from .proto_definitions import features as PROTO_FEATURES

# Parse records


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class FeatureProto(object):

    # Reading the data
    features = PROTO_FEATURES



    parser_proto = {}
    for prototype in features:
        feat_name = prototype.name
        dtype = prototype.dtype
        shape = prototype.shape
        parser_proto[feat_name] = tf.FixedLenFeature(() if shape == 1 else (shape), dtype)

    def __init__(self, one_hot=False):
        self.one_hot = one_hot

    @property
    def size(self):
        size = 0
        for prototype in self.features:
            size += prototype.shape
        return size

    def dataset_creation(self, data):
        idx = 0
        collection = {}
        for prototype in self.features:
            datum = [data[idx]] if prototype.shape == 1 else data[idx:idx + prototype.shape]
            encoded_feature = self._generate_feature(
                prototype.dtype, datum, idx
            )
            collection[prototype.name] = encoded_feature
            idx += prototype.shape
        return collection

    def _generate_feature(self, dtype, data, idx):
        if dtype == tf.float16 or dtype == tf.float32 or dtype == tf.float64:
            encoded_feature = _float_feature(data)
        elif dtype == tf.int16 or dtype == tf.int32 or dtype == tf.int64:
            encoded_feature = _int64_feature(data)
        elif dtype == tf.string:
            encoded_feature = _bytes_feature(data)
        else:
            raise NotImplementedError('Unmated type while generating feature in FeatureProto')
        return encoded_feature

    def unpack(self, example_proto):
        features = self.parser_proto
        parsed_features = tf.parse_single_example(example_proto, features)
        for k, v in parsed_features.items():
            parsed_features[k] = tf.cast(v, dtype=tf.int32)
        labels = parsed_features['Wilderness_Area']
        parsed_features.pop('Wilderness_Area')

        # Then, convert the dataset into tensors which tensorflow expects?
        parsed_features['Soil_Type'] = tf.convert_to_tensor(parsed_features['Soil_Type'])

        # Label managing
        labels = tf.cast(tf.argmax(labels, axis=0), dtype=tf.int32)

        if self.one_hot:
            labels = tf.one_hot(tf.cast(labels, dtype=tf.uint8), 8, on_value=1, off_value=0, axis=-1)
        return parsed_features, labels


    
    def get_feature_columns(self):
        """
        feature_columns: An iterable containing the FeatureColumns to use as inputs to your model. 

        All items should be instances of classes derived from _DenseColumn such as numeric_column, embedding_column, bucketized_column, indicator_column. 
        If you have categorical features, you can wrap them with an embedding_column or indicator_column
        
        E.g:
            price = numeric_column('price')
            keywords_embedded = embedding_column(
                categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
            columns = [price, keywords_embedded, ...]
        """
        return [v.feature_column for v in self.features if v.feature_column is not None]


def dataset_config(repeat=False, batch_size=32, num_cpus=None, return_dataset=False, shuffle=True,
                   # Used in tfRecordDatasets
                   filenames: list = None, mapper=None,
                   # Used in from_tensor_slices
                   initializable: Tuple = False, sess=None, feed_dict=None):
    """
    Supports 2 modes: from tensor_slices OR from tfRecordDatasets
    """
    tf_record = mapper is not None and filenames is not None
    tensor_slices = initializable is not None and sess is not None and feed_dict is not None

    if tf_record:
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(mapper, num_parallel_calls=num_cpus)
    elif tensor_slices:
        assert initializable is not False, 'initializable should be an iterable with placeholders'
        dataset = tf.data.Dataset.from_tensor_slices(initializable)
    else:
        raise ValueError('If loading from tfRecordDatasets fill in filenames and mapper. '
                         'If using from_tensor_slices feed in a initializable(placeholder iterable), session, and feed_dict')

    if repeat:  # For epochs
        dataset = dataset.repeat(repeat)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)

    if return_dataset:  # Useful in canned estimators which can work directly
        return dataset

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels


def split_into_linear_numeric(features):
    if isinstance(features, list):  # Used when
        linear = features[-2:]
        numeric = features[:-2]
        return linear, numeric
    assert isinstance(features, dict), 'Unsupported type in split_into_linear_numeric: {}'.format(type(features))

    # Hard-coded to be fast

    linear = {'Soil_Type': features['Soil_Type'],
              'Cover_Type': features['Cover_Type']}

    features.pop('Soil_Type')
    features.pop('Cover_Type')

    return linear, features