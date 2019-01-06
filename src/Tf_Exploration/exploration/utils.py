"""
utils.py
    - for use with tf_exploration. Contains classes from Notebook1-data_exploration (for use in Notebook2)
    
    - lets us focus on notebook 2. Will serve as jumping point for the production series
"""

# Dataset creation
# Dataset reading
# placeholder definition

import tensorflow as tf

# Parse records

def _bytes_feature(value, shape):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value, shape):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value, shape):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

class FeatureProto(object):
    from collections import namedtuple
    
    proto = namedtuple('prototype', ['name', 'dtype', 'shape'])
    
    features = [
        proto(name='Elevation', dtype=tf.float32, shape=1),
        proto(name='Aspect', dtype=tf.float32, shape=1),
        proto(name='Slope', dtype=tf.float32, shape=1),
        proto(name='Horizontal_Distance_To_Hydrology', dtype=tf.float32, shape=1),
        proto(name='Vertical_Distance_To_Hydrology', dtype=tf.float32, shape=1),
        proto(name='Horizontal_Distance_To_Roadways', dtype=tf.float32, shape=1),
        proto(name='Hillshade_9am', dtype=tf.float32, shape=1),
        proto(name='Hillshade_Noon', dtype=tf.float32, shape=1),
        proto(name='Hillshade_3pm', dtype=tf.float32, shape=1),
        proto(name='Horizontal_Distance_To_Fire_Points', dtype=tf.float32, shape=1),
        proto(name='Wilderness_Area', dtype=tf.float32, shape=4),
        proto(name='Soil_Type', dtype=tf.float32, shape=40),
        proto(name='Cover_Type', dtype=tf.float32, shape=1),
    ]
    
    
    def dataset_creation(self, data):
        idx = 0
        collection = {}
        for prototype in self.features:
            encoded_feature = self._generate_feature(
                prototype.dtype, prototype.shape, data, idx
            )
            collection[prototype.name] = encoded_feature
            idx += prototype.shape
        return collection

    def _generate_feature(self, dtype, shape, data, idx):
        datum = [data[idx]] if shape == 1 else data[idx:idx + shape]

        if dtype == tf.float16 or dtype == tf.float32 or dtype == tf.float64:
            encoded_feature = _float_feature(datum, shape)
        elif dtype == tf.int16 or dtype == tf.int32 or dtype == tf.int64:
            encoded_feature = _int64_feature(datum, shape)
        elif dtype == tf.string:
            encoded_feature = _bytes_feature(datum, shape)
        else:
            raise NotImplementedError('Unmated type while generating feature in FeatureProto')
        return encoded_feature
    
    def _dataset_parsing(self):
        if hasattr(self, 'parser_proto'):
            return self.parser_proto
        else:
            parser_proto = {}
            for prototype in self.features:
                feat_name = prototype.name
                dtype = prototype.dtype
                shape = prototype.shape
                parser_proto[feat_name] = tf.FixedLenFeature(() if shape == 1 else (shape), dtype)
            self.parser_proto = parser_proto
            return self.parser_proto
        
    def unpack(self, example_proto):
        features = self._dataset_parsing()
        parsed_features = tf.parse_single_example(example_proto, features)
        labels = parsed_features['Cover_Type']
        parsed_features.pop('Cover_Type')
        # Then, convert the dataset into tensors which tensorflow expects?
        parsed_features['Soil_Type'] = tf.convert_to_tensor(parsed_features['Soil_Type'])
        parsed_features['Wilderness_Area'] = tf.cast(tf.argmax(parsed_features['Wilderness_Area'], axis=0), dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int32)
        #labels = tf.one_hot(tf.cast(labels, dtype=tf.uint8), 8, on_value=1, off_value=0, axis=-1)

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
        label_less = self.features
        held = []
        label = 'Cover_Type'
        for v in label_less:
            if v.name == 'Cover_Type':
                continue
            else:
                if v.name != 'Soil_Type':
                    held.append(tf.feature_column.numeric_column(v.name))
                else:
                    held.append(tf.feature_column.numeric_column(v.name, shape=(40)))
        return held

def dataset_config(filenames: list, mapper=None, repeat=False, batch_size=32,
                  initializable=False, sess=None, feed_dict=None, num_cpus=None):
    dataset = tf.data.TFRecordDataset(filenames)
    
    if mapper is not None:
        dataset = dataset.map(mapper, num_parallel_calls=num_cpus)
        
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)
    

    iterator = dataset.make_one_shot_iterator()
        
    if initializable:
        assert feed_dict is not None, 'Supply feed dict to initializable iterator'
        sess.run(iterator.initializer, feed_dict=feed_dict)
    
    next_element = iterator.get_next()
    return next_element