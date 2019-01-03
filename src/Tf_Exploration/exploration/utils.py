"""
utils.py
    - for use with tf_exploration. Contains classes from Notebook1-data_exploration (for use in Notebook2)
    
    - lets us focus on notebook 2. Will serve as jumping point for the production series
"""

# Dataset creation
# Dataset reading
# placeholder definition

import tensorflow as tf

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
            feat_name = prototype.name
            dtype = prototype.dtype
            shape = prototype.shape
            
            if dtype == tf.float32:
                if shape == 1:
                    datum = data[idx]
                    encoded_feature = _float_feature(datum)
                else:
                    datum = data[idx: idx+shape]
                    encoded_feature = _tensor_feature(datum, 'float_list', tf.train.FloatList)
            else:
                raise NotImplementedError('dataset creation for non-float32 not supported')
            
            collection[feat_name] = encoded_feature
            idx += shape
        return collection
    
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
    
    if initializable:
        """
        An initializable iterator requires you to run an explicit iterator.initializer operation before using it. 
        In exchange for this inconvenience, it enables you to parameterize the definition of the dataset, 
        using one or more tf.placeholder() tensors that can be fed when you initialize the iterator
        """
        # Creates an Iterator for enumerating the elements of this dataset
        if sess is None:
            raise Exception('Initializable dataset configuration specified but session not supplied')
        iterator = dataset.make_initializable_iterator()
    else:
        """
        A one-shot iterator is the simplest form of iterator, which only supports iterating once through a dataset, 
        with no need for explicit initialization. One-shot iterators handle almost all of the cases that the existing 
        queue-based input pipelines support, but they do not support parameterization
        """
        iterator = dataset.make_one_shot_iterator()
        
    if initializable:
        assert feed_dict is not None, 'Supply feed dict to initializable iterator'
        sess.run(iterator.initializer, feed_dict=feed_dict)
    
    next_element = iterator.get_next()
    return next_element