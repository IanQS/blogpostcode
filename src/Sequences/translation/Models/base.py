import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from enum import Enum
import logging

class EmbeddingStates(Enum):
    UNINITIALIZED = -1
    NONE = 0
    HUB = 1
    LOCAL = 2


class TranslatorBase(object):
    def __init__(self, top_k, word_index, embedding_path, embedding_dim):
        self.logger = logging.getLogger(__name__)
        self.top_k = top_k
        self.word_index = word_index
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.embedding_state = EmbeddingStates.UNINITIALIZED

    def get_embedding_matrix(self):
        # Read the pre-trained embedding file and get word to word vector mappings.

        if self.embedding_path is None:
            self.embedding_state = EmbeddingStates.NONE
            embedding = None
        elif 'tfhub' in self.embedding_path:
            self.embedding_state = EmbeddingStates.HUB
            embedding = self._get_tfhub_embedding()
        else:
            self.embedding_state = EmbeddingStates.LOCAL
            embedding = self._get_local_embedding()

        self.logger.debug(self.embedding_state)
        return embedding

    def _get_tfhub_embedding(self):
        with tf.Graph().as_default():
            embed = hub.Module(self.embedding_path)
            embeddings = embed(self.word_index)
            return embeddings


    def _get_local_embedding(self):
        embedding_matrix_all = {}
        with open(self.embedding_path) as f:
            for line in f:  # Every line contains word followed by the vector value
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_matrix_all[word] = coefs

        # Prepare embedding matrix with just the words in our word_index dictionary
        num_words = min(len(self.word_index) + 1, self.top_k)
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        for word, i in self.word_index.items():
            if i >= self.top_k:
                continue
            embedding_vector = embedding_matrix_all.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
