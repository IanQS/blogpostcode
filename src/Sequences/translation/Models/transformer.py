from Sequences.translation.Models.base import TranslatorBase
import logging

class Transformer(TranslatorBase):
    def __init__(self, sess, model_hparams, embedding):
        self.logger = logging.getLogger(__name__)
        self.logger.debug(''.format(model_hparams))
        self.logger.debug('Base args: {}'.format(embedding))
        self.sess = sess
        self.hparams = model_hparams
        super().__init__(**embedding)