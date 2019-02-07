"""
Takes embedding for generic vocabulary and extracts the embeddings
  matching the current vocabulary
  The pre-trained embedding file is obtained from https://nlp.stanford.edu/projects/glove/
  # Arguments: 
      word_index: dict, {key =word in vocabulary: value= integer mapped to that word}
      embedding_path: string, location of the pre-trained embedding file on disk
      embedding_dim: int, dimension of the embedding space
  # Returns: numpy matrix of shape (vocabulary, embedding_dim) that contains the embedded
      representation of each word in the vocabulary.
"""
def get_embedding_matrix(word_index, embedding_path, embedding_dim):
    # Read the pre-trained embedding file and get word to word vector mappings.
    embedding_matrix_all = {}

    # Download if embedding file is in GCS
    if embedding_path.startswith('gs://'):
        download_from_gcs(embedding_path, destination='embedding.csv')
        embedding_path = 'embedding.csv'

    with open(embedding_path) as f:
        for line in f:  # Every line contains word followed by the vector value
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix_all[word] = coefs

    # Prepare embedding matrix with just the words in our word_index dictionary
    num_words = min(len(word_index) + 1, TOP_K)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i >= TOP_K:
            continue
        embedding_vector = embedding_matrix_all.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


"""
Main orchestrator. Responsible for calling all other functions in model.py
  # Arguments: 
      output_dir: string, file path where training files will be written
      hparams: dict, command line parameters passed from task.py
  # Returns: nothing, kicks off training and evaluation
"""
def train_and_evaluate(output_dir, hparams):
    # Load Data
    ((train_texts, train_labels), (test_texts, test_labels)) = load_hacker_news_data(
        hparams['train_data_path'], hparams['eval_data_path'])

    # Create vocabulary from training corpus.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Generate vocabulary file from tokenizer object to enable
    # creating a native tensorflow lookup table later (used in vectorize_sentences())
    tf.gfile.MkDir(output_dir) # directory must exist before we can use tf.gfile.open
    global VOCAB_FILE_PATH; VOCAB_FILE_PATH = os.path.join(output_dir,'vocab.txt')
    with tf.gfile.Open(VOCAB_FILE_PATH, 'wb') as f:
        f.write("{},0\n".format(PADWORD))  # map padword to 0
        for word, index in tokenizer.word_index.items():
            if index < TOP_K: # only save mappings for TOP_K words
                f.write("{},{}\n".format(word, index))

    # Create estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
    estimator = keras_estimator(
        model_dir=output_dir,
        config=run_config,
        learning_rate=hparams['learning_rate'],
        embedding_path=hparams['embedding_path'],
        word_index=tokenizer.word_index
    )

    # Create TrainSpec
    train_steps = hparams['num_epochs'] * len(train_texts) / hparams['batch_size']
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:input_fn(
            train_texts,
            train_labels,
            hparams['batch_size'],
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=train_steps
    )

    # Create EvalSpec
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:input_fn(
            test_texts,
            test_labels,
            hparams['batch_size'],
            mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        exporters=exporter,
        start_delay_secs=10,
        throttle_secs=10
    )

    # Start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)