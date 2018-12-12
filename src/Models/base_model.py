"""
base_model
    - ABC for any models that are implemented. Any generic functions that should be common to most/all
    NLP models should be inherited from here

Author: Ian Q.

Notes:
"""


import tensorflow as tf
from abc import abstractmethod, ABC


class BaseModel(ABC):
    def __init__(self, sess: tf.InteractiveSession, graph: tf.Graph,
                 debug: bool, model_name: str, learning_rate=0.01):
        self.sess = sess
        self.graph = graph
        self.debug = debug
        self.model_name = model_name
        self.lr = learning_rate

        with tf.variable_scope(model_name):
            self.outputs = self.construct_layers()
            self.cost, self.train_op = self.construct_training_method()

        self.file_writer = tf.summary.FileWriter('.', self.sess.graph)

        self.summ = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    ################################################
    # Public Interfaces
    ################################################

    def initialize(self, load_path=None):
        if load_path is not None:
            self.saver.restore(self.sess, load_path)
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def summary_path(self):
        return '.'

    def construct_layers(self, **layer_params):
        return self._construct_layers(**layer_params)

    def save(self, save_path: str) -> None:
        self.saver.save(self.sess, save_path)

    def train(self, input_seq, labels):
        return self.sess.run([self.summ, self.train_op], feed_dict={self.input_seq: input_seq, self.labels: labels})

    def predict(self, input_seq):
        return self.sess.run([self.outputs], feed_dict={self.input_seq: input_seq})

    def construct_training_method(self):
        cost = self._construct_training_method()

        assert cost is not None

        tf.summary.scalar('cost', cost)
        optimizer = tf.train.AdamOptimizer(self.lr)

        grads = optimizer.compute_gradients(cost)
        # Update the weights wrt to the gradient
        optimizer = optimizer.apply_gradients(grads)
        # Save the grads with tf.summary.histogram
        for index, grad in enumerate(grads):
            tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

        return cost, optimizer

    ################################################
    # Private Interfaces
    #   - you need to override these
    ################################################


    @abstractmethod
    def _construct_layers(self, **layer_params):
        raise NotImplementedError

    @abstractmethod
    def _construct_training_method(self):
        raise NotImplementedError
