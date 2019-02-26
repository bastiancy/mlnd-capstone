from pathlib import Path
import functools
import json
import random
import numpy as np
import tensorflow as tf

# Global config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('line', None, 'The exaple text to process.')
flags.DEFINE_string('model_dir', './results/ned/lc-8', 'Folder where the model checkpoint is located.')
flags.DEFINE_string('params_file', './results/ned/params.json', 'Path to the parameters file.')
formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'END': '\033[0m',
}

from train import model_fn, fwords, ftags


def pretty_print(line, preds, golds=None):
    words = line.strip().split()
    lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
    padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
    padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
    print('words : {}'.format(' '.join(padded_words)))

    if golds is None:
        values = formatters
        values['TEXT'] = ' '.join(padded_preds)
        print('preds : {YELLOW}{TEXT}{END}'.format(**values))
    else:
        golds = golds.strip().split()
        lengths = [max(len(w), len(p)) for w, p in zip(words, golds)]
        padded_golds = [w + (l - len(w)) * ' ' for w, l in zip(golds, lengths)]
        print('labels: {}'.format(' '.join(padded_golds)))

        inter = np.in1d(padded_preds, padded_golds)
        for idx, isinter in enumerate(inter):
            padded_preds[idx] = '{GREEN}' + padded_preds[idx]  + '{END}' if isinter else '{RED}' + padded_preds[idx]  + '{END}'
            padded_preds[idx] = padded_preds[idx].format(**formatters)
        print('preds : {}'.format(' '.join(padded_preds)))


def predict_input_fn(line):
    # Words
    words = [w.encode() for w in line.strip().split()]
    nwords = len(words)

    # Wrapping in Tensors
    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)

    return (words, nwords), None


def main(_):
    with Path(FLAGS.params_file).open() as f:
        params = json.load(f)

    gold = None
    if FLAGS.line is None:
        with Path(fwords('testb')).open('r') as f_words, \
                Path(ftags('testb')).open('r') as f_tags:
            line_words = f_words.readlines()
            line_tags = f_tags.readlines()
        idx = random.randint(0, len(line_words) - 1)
        line = line_words[idx]
        gold = line_tags[idx]
    else:
        line = FLAGS.line

    params['words'] = str(Path(FLAGS.data_dir, 'vocab.words.txt'))
    params['chars'] = str(Path(FLAGS.data_dir, 'vocab.chars.txt'))
    params['tags'] = str(Path(FLAGS.data_dir, 'vocab.tags.txt'))
    params['glove'] = str(Path(FLAGS.data_dir, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, FLAGS.model_dir, params=params)
    predict_inpf = functools.partial(predict_input_fn, line)
    for pred in estimator.predict(predict_inpf):
        pretty_print(line, pred['tags'], gold)
        break


if __name__ == '__main__':
  tf.app.run()