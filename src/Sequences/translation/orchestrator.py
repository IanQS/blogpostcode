from tensorflow.python.keras.preprocessing import text
import tensorflow as tf

from Sequences.translation.Logger.logger import logging_setup
from Sequences.translation.Models.architectures import model_builder
from Sequences.translation.Pipeline.dataset import Dataset

class Orchestrator(object):
    def __init__(self, sess, model_config:dict, dataset_config:dict):
        logging_setup()
        self.sess = sess
        self.ds = Dataset(**dataset_config)
        self.model = model_builder(**model_config)


    def train_and_evaluate(self):
        tokenizer = text.Tokenizer(num)

# def train_and_evaluate(output_dir, ):
#     # Load Data
#     ((train_texts, train_labels), (test_texts, test_labels)) = load_hacker_news_data(
#         hparams['train_data_path'], hparams['eval_data_path'])

#     # Create vocabulary from training corpus.
#     tokenizer = text.Tokenizer(num_words=TOP_K)
#     tokenizer.fit_on_texts(train_texts)

#     # Generate vocabulary file from tokenizer object to enable
#     # creating a native tensorflow lookup table later (used in vectorize_sentences())
#     tf.gfile.MkDir(output_dir) # directory must exist before we can use tf.gfile.open
#     global VOCAB_FILE_PATH; VOCAB_FILE_PATH = os.path.join(output_dir,'vocab.txt')
#     with tf.gfile.Open(VOCAB_FILE_PATH, 'wb') as f:
#         f.write("{},0\n".format(PADWORD))  # map padword to 0
#         for word, index in tokenizer.word_index.items():
#             if index < TOP_K: # only save mappings for TOP_K words
#                 f.write("{},{}\n".format(word, index))

#     # Create estimator
#     run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
#     estimator = Transformer(**model_params)

#     # Create TrainSpec
#     train_steps = hparams['num_epochs'] * len(train_texts) / hparams['batch_size']
#     train_spec = tf.estimator.TrainSpec(
#         input_fn=lambda:input_fn(
#             train_texts,
#             train_labels,
#             hparams['batch_size'],
#             mode=tf.estimator.ModeKeys.TRAIN),
#         max_steps=train_steps
#     )

#     # Create EvalSpec
#     exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
#     eval_spec = tf.estimator.EvalSpec(
#         input_fn=lambda:input_fn(
#             test_texts,
#             test_labels,
#             hparams['batch_size'],
#             mode=tf.estimator.ModeKeys.EVAL),
#         steps=None,
#         exporters=exporter,
#         start_delay_secs=10,
#         throttle_secs=10
#     )

#     # Start training
#     tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    from Sequences.translation.Pipeline.config import LOAD_LOC, SAVE_LOC_RECORDS, pattern, DATASET_DEFAULTS, SAVE_LOC_NPY
    
    use_raw = True
    sess = tf.InteractiveSession()
    
    MODEL_CONFIG = {
        'name': 'Transformer'
    }
    
    DATASET_CONFIG = {
        'load_loc': LOAD_LOC,
        'save_loc': SAVE_LOC_NPY if use_raw else SAVE_LOC_RECORDS,
        'pattern': pattern,
        'sess': sess,
        'use_raw': use_raw   
    }
    
    orchestrator = Orchestrator(sess, MODEL_CONFIG, DATASET_CONFIG)