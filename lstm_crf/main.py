"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys
import random
import time

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

DATADIR = '../dataset/conll2002/esp'
OUTDIR = './results/esp'

# Logging
Path(OUTDIR).mkdir(exist_ok=True, parents=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler(str(Path(OUTDIR, 'main.log'))),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def fwords(name, folder=DATADIR):
    return str(Path(folder, '{}.words.txt'.format(name)))


def ftags(name, folder=DATADIR):
    return str(Path(folder, '{}.tags.txt'.format(name)))


def fscores(name, folder=DATADIR):
    return str(Path(folder, '{}.scores.txt'.format(name)))


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


def input_fn(words, tags, params=None, shuffle=False, repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle:
        dataset = dataset.shuffle(params['buffer'])

    if repeat:
        dataset = dataset.repeat(params['epochs'])

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
    mnlp = (best_score / tf.to_float(nwords))

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings,
            'score': mnlp
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
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


def run_train(train_inpf, eval_inpf, run_dir, params=None, warm_dir=None):
    """ Estimator, train and evaluate"""
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=20)

    if warm_dir is not None:
        estimator = tf.estimator.Estimator(model_fn, run_dir, cfg, params, warm_dir)
    else:
        estimator = tf.estimator.Estimator(model_fn, run_dir, cfg, params)

    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=1000, run_every_secs=10)

    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=10)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator


def save_pool_predictions(estimator, test_inpf):
    with Path(OUTDIR, 'pool.scores.txt').open('wb') as f1:
        preds_gen = estimator.predict(test_inpf)
        for preds in preds_gen:
            f1.write(str(preds['score']).encode() + b'\n')


if __name__ == '__main__':
    start = time.time()
    random.seed(42)
    tf.random.set_random_seed(42)

    # Hyperparameters
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 10000,
        'lstm_size': 100,
        'sampling': 'mnlp',
        'sample_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz')),
    }

    run_dir = str(Path(OUTDIR, '{}-0'.format(params['sampling'])))
    Path(run_dir).mkdir(parents=True, exist_ok=False)

    with Path(OUTDIR, 'params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    # # Copy 1% of "train" dataset to run-dir
    from shutil import copyfile

    copyfile(fwords('train1'), fwords('train', folder=OUTDIR))
    copyfile(ftags('train1'), ftags('train', folder=OUTDIR))

    # Start the model, train it with "results/train" dataset (1% of original), and evaluate on "testa" dataset.
    train_inpf = functools.partial(input_fn, fwords('train', folder=OUTDIR), ftags('train', folder=OUTDIR), params)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))
    estimator = run_train(train_inpf, eval_inpf, run_dir, params=params)

    # there is not need to save predictions, because rand sampling does not use the scores
    if params['sampling'] != 'rand':
        # Make predictions on the pool dataset, then save the scores
        test_inpf = functools.partial(input_fn, fwords('train'), ftags('train'))
        save_pool_predictions(estimator, test_inpf)

    # Run 10 online passes to sample from the pool
    for i in range(1, 5):
        run_dir = str(Path(OUTDIR, '{}-{}'.format(params['sampling'], i)))
        Path(run_dir).mkdir(parents=True, exist_ok=False)

        # load pool data to choose examples from
        with Path(fwords('train')).open('r') as f_words, \
                Path(ftags('train')).open('r') as f_tags:
            line_words = f_words.readlines()
            line_tags = f_tags.readlines()

        if params['sampling'] == 'rand':
            # choose indices randomly
            indices = list(range(len(line_words)))
            random.shuffle(indices)

        elif params['sampling'] == 'mnlp':
            # load scores from pool and sort lines ascending, then extract the indices.
            # those indices will be used to choose the new examples from pool
            with Path(fscores('pool', folder=OUTDIR)).open('r') as f_scores:
                line_scores = f_scores.readlines()

            # load scores file and sort lines ascending, then extract the indices.
            indices = argsort(line_scores)

        else:
            raise RuntimeError('sampling method "{}" is invalid!'.format(params['sampling']))

        # add new examples to the train dataset
        with Path(fwords('train', folder=OUTDIR)).open('a') as f_words, \
                Path(ftags('train', folder=OUTDIR)).open('a') as f_tags:
            # Max budget for sampling
            indices = indices[: params['sample_size']]

            # append selected samples to dataset
            for idx in indices:
                f_words.write(line_words[idx])
                f_tags.write(line_tags[idx])

        # Use the updated "train" dataset to retrain the model
        train_inpf = functools.partial(input_fn, fwords('train', folder=OUTDIR), ftags('train', folder=OUTDIR), params,
                                       shuffle=True, repeat=True)
        estimator = run_train(train_inpf, eval_inpf, run_dir, params=params)

        # there is not need to save predictions, because rand sampling does not use the scores
        if params['sampling'] != 'rand':
            # Again, make predictions on the pool dataset with the updated model then save the scores.
            test_inpf = functools.partial(input_fn, fwords('train'), ftags('train'))
            save_pool_predictions(estimator, test_inpf)

    print("\n>> Finished in {} secs\n".format(time.time() - start))
