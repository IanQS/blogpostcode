from collections import namedtuple
import tensorflow as tf
proto = namedtuple('prototype', ['name', 'dtype', 'shape', 'feature_column'])

features = [
    proto(name='Elevation', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Elevation')),

    proto(name='Aspect', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Aspect')),

    proto(name='Slope', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Slope')),

    proto(name='Horizontal_Distance_To_Hydrology', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Horizontal_Distance_To_Hydrology')),

    proto(name='Vertical_Distance_To_Hydrology', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Vertical_Distance_To_Hydrology')),

    proto(name='Horizontal_Distance_To_Roadways', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Horizontal_Distance_To_Roadways')),

    proto(name='Hillshade_9am', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Hillshade_9am')),

    proto(name='Hillshade_Noon', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Hillshade_Noon')),

    proto(name='Hillshade_3pm', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Hillshade_3pm')),

    proto(name='Horizontal_Distance_To_Fire_Points', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.numeric_column('Horizontal_Distance_To_Fire_Points')),

    proto(name='Wilderness_Area', dtype=tf.float32, shape=4,
          feature_column=None),

    proto(name='Soil_Type', dtype=tf.float32, shape=40,
          feature_column=tf.feature_column.numeric_column('Soil_Type', shape=(40,))),

    #Used in the tutorial
    proto(name='Cover_Type', dtype=tf.float32, shape=1,
          feature_column=tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_identity('Cover_Type', num_buckets=8),
              dimension=10
          )
    ),
    # proto(name='Cover_Type', dtype=tf.float32, shape=1,
    #       feature_column=tf.feature_column.categorical_column_with_identity('Cover_Type', num_buckets=8))
]