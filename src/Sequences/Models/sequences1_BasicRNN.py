from Sequences.Models.base_model import BaseModel

import tensorflow as tf

class SimpleRNN(BaseModel):
    def __init__(self, sess: tf.Session, graph: tf.Graph, debug: bool):
        super().__init__(sess, graph, debug, 'SimpleRNN')

if __name__ == '__main__':
    ################################################
    #Dataset and processor initialization and config
    ################################################
    from Sequences.DataPipelines.data_backend import Pipeline
    from Sequences.DataPipelines.common_preprocessors import log_normalizations
    from Sequences.DataPipelines.common_backends import np_backend

    pipeline_config = {
        'backend': 'Folder',
        'processed': True,
        'processor': log_normalizations,
        'connect_to_backend': np_backend
    }

    dataset = Pipeline(pipeline_config)

    ################################################
    #Model initialization and config
    ################################################
    sess = tf.Session()
    graph = tf.Graph()
    debug = True
    model = SimpleRNN(sess, graph, debug)

