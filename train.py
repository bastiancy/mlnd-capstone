import functools
import json
import logging
import sys
import random
import time
from pathlib import Path
from shutil import copyfile
import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

# Global config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './dataset/conll2002/ned', 'Folder containing the formated dataset (see convert.py).')
flags.DEFINE_string('out_dir', './results/ned', 'Folder where the model, logs, and tensorboard metrics will be saved.')
flags.DEFINE_string('sampling', 'lc', 'Sampling mode, "lc" or "rand".')
flags.DEFINE_integer('run_counts', 2, 'Ho many times we want to do retrain for the Online Learning.')
flags.DEFINE_boolean('verbose', True, 'Wheter print tensorflow logs to console.')
SAMPLING_MODES = ['rand', 'lc']


def fwords(name, folder=FLAGS.data_dir):
    return str(Path(folder, '{}.words.txt'.format(name)))


def ftags(name, folder=FLAGS.data_dir):
    return str(Path(folder, '{}.tags.txt'.format(name)))


def fscores(name, folder=FLAGS.data_dir):
    return str(Path(folder, '{}.scores.txt'.format(name)))


def init_train_dataset(max_size=0.01):
    # use the same file if already exist
    if not Path(fwords('train1', folder=FLAGS.out_dir)).is_file():
        with Path(fwords('train')).open('r') as f_words, \
                Path(ftags('train')).open('r') as f_tags:
            line_words = f_words.readlines()
            line_tags = f_tags.readlines()

        # Choose indices randomly
        indices = list(range(len(line_words) - 1))
        random.shuffle(indices)
        if 1 > max_size > 0:
            indices = indices[: int(max_size * len(indices))]

        # add new examples to the train dataset
        with Path(fwords('train1', folder=FLAGS.out_dir)).open('w') as f_words, \
                Path(ftags('train1', folder=FLAGS.out_dir)).open('w') as f_tags:
            # append selected samples to dataset
            for j, index in enumerate(indices):
                f_words.write(line_words[index].strip())
                f_tags.write(line_tags[index].strip())
                if j < len(indices) - 1:
                    f_words.write("\n")
                    f_tags.write("\n")

    copyfile(fwords('train1', folder=FLAGS.out_dir), fwords('train', folder=FLAGS.out_dir))
    copyfile(ftags('train1', folder=FLAGS.out_dir), ftags('train', folder=FLAGS.out_dir))


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    embeddings = tf.nn.embedding_lookup(variable, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, best_score = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }

        # Score used for active learning
        if params['sampling'] == 'lc':
            predictions['score'] = tf.to_float(1.0) - best_score

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


def run_train(train_inpf, eval_inpf, run_dir, params=None, warm_start_from=None):
    """ Estimator, train and evaluate"""
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=params['train_save_checkpoints_steps'],
                                 save_summary_steps=params['train_save_summary_steps'],
                                 log_step_count_steps=params['train_log_step_count_steps'])
    estimator = tf.estimator.Estimator(model_fn, run_dir, cfg, params, warm_start_from=warm_start_from)

    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', params['stop_max_steps'], run_every_secs=params['stop_run_every_secs'])

    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=params['eval_throttle_secs'])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator


def save_pool_predictions(estimator, test_inpf):
    with Path(FLAGS.out_dir, 'pool.scores.txt').open('wb') as f1:
        preds_gen = estimator.predict(test_inpf)
        for preds in preds_gen:
            f1.write(str(preds['score']).encode() + b'\n')


def main(_):
    # Logging
    Path(FLAGS.out_dir).mkdir(exist_ok=True, parents=True)
    if FLAGS.verbose:
        tf.logging.set_verbosity(logging.INFO)
    else:
        tf.logging.set_verbosity(logging.ERROR)
    handlers = [
        logging.FileHandler(str(Path(FLAGS.out_dir, 'main.log'))),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

    random.seed(42)
    tf.random.set_random_seed(42)
    assert FLAGS.sampling in SAMPLING_MODES, 'Sampling method is invalid!'
    assert FLAGS.run_counts >= 2, 'The minimum for runs counts is 2!'

    start = time.time()
    run_times = []

    # Hyperparameters
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 1,
        'batch_size': 80,
        'buffer': 10000,
        'lstm_size': 100,
        'sampling': FLAGS.sampling,
        'sample_size': 800,
        'words': str(Path(FLAGS.data_dir, 'vocab.words.txt')),
        'chars': str(Path(FLAGS.data_dir, 'vocab.chars.txt')),
        'tags': str(Path(FLAGS.data_dir, 'vocab.tags.txt')),
        'glove': str(Path(FLAGS.data_dir, 'glove.npz')),
        'train_save_checkpoints_steps': 10,
        'train_save_summary_steps': 10,
        'train_log_step_count_steps': 10,
        'stop_max_steps': 200,
        'stop_run_every_secs': 20,
        'eval_throttle_secs': 0,
    }

    run_dir = str(Path(FLAGS.out_dir, '{}-0'.format(params['sampling'])))
    Path(run_dir).mkdir(parents=True, exist_ok=False)

    # Use 1% of training dataset to start the model
    init_train_dataset()

    # Start the model, train it with "results/train" dataset (1% of original), and evaluate on "testa" dataset.
    train_inpf = functools.partial(input_fn, fwords('train', folder=FLAGS.out_dir), ftags('train', folder=FLAGS.out_dir), params)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))
    estimator = run_train(train_inpf, eval_inpf, run_dir, params=params)

    # adjust params
    params['batch_size'] = 160
    params['epochs'] = 5
    with Path(FLAGS.out_dir, 'params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    # Here we run multiple passes of the online learning.
    # Each time data from the pool will be choosen, and the model will be retrain on the updated dataset
    for i in range(1, FLAGS.run_counts):
        run_name = '{}-{}'.format(params['sampling'], i)
        run_dir = str(Path(FLAGS.out_dir, run_name))
        Path(run_dir).mkdir(parents=True, exist_ok=False)
        start_run = time.time()

        # When using uncertainty sampling we need to make predictions on the pool dataset, with the previous state
        # of the model. On the contrary, random sampling does not need the scores, so prediction is not required.
        if params['sampling'] in ['lc']:
            # Make predictions on the pool dataset, then save the scores in a temp file.
            test_inpf = functools.partial(input_fn, fwords('train'), ftags('train'))
            save_pool_predictions(estimator, test_inpf)

        # load pool data to choose examples from
        with Path(fwords('train')).open('r') as f_words, \
                Path(ftags('train')).open('r') as f_tags:
            line_words = f_words.readlines()
            line_tags = f_tags.readlines()

        if params['sampling'] in ['lc']:
            # Load scores calculated on pool. We will peek examples from the pool based on the score, then we
            # append those examples to the train dataset.
            line_scores = np.loadtxt(fscores('pool', folder=FLAGS.out_dir))
            # Sort ascending the scores list, then extract the indices
            indices = np.argsort(line_scores)
        elif params['sampling'] == 'rand':
            # Choose indices randomly
            indices = list(range(len(line_words) - 1))
            random.shuffle(indices)
        else:
            raise RuntimeError('sampling method "{}" is invalid!'.format(params['sampling']))

        # add new examples to the train dataset
        with Path(fwords('train', folder=FLAGS.out_dir)).open('a') as f_words, \
                Path(ftags('train', folder=FLAGS.out_dir)).open('a') as f_tags:
            # Max budget for sampling
            indices = indices[: params['sample_size']]

            # append selected samples to dataset
            for j, index in enumerate(indices):
                f_words.write("\n" + line_words[index].strip())
                f_tags.write("\n" + line_tags[index].strip())

        # Use the updated "train" dataset to retrain the model.
        # We use the same validation dataset as before.
        train_inpf = functools.partial(input_fn,
                                       fwords('train', folder=FLAGS.out_dir),
                                       ftags('train', folder=FLAGS.out_dir),
                                       params, shuffle_and_repeat=True)

        # We want to incrementaly update de model so we warm it up from the latest checkpoint
        warm_dir = str(Path(FLAGS.out_dir, '{}-{}'.format(params['sampling'], i - 1)))
        estimator = run_train(train_inpf, eval_inpf, run_dir, params=params, warm_start_from=warm_dir)

        run_times.append((run_name, time.time() - start_run))
        copyfile(fwords('train', folder=FLAGS.out_dir), fwords(run_name + '.train', folder=FLAGS.out_dir))
        copyfile(ftags('train', folder=FLAGS.out_dir), ftags(run_name + '.train', folder=FLAGS.out_dir))
        if Path(fscores('pool', folder=FLAGS.out_dir)).is_file():
            copyfile(fscores('pool', folder=FLAGS.out_dir), fscores(run_name + '.pool', folder=FLAGS.out_dir))

    print("\n>> Finished in {:.3f} secs".format(time.time() - start))
    for (name, timing) in run_times:
        print(">> Run {} took {:.3f} secs".format(name, timing))


if __name__ == '__main__':
  tf.app.run()