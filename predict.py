"""Reload and serve a saved model"""

import random
from pathlib import Path
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Folder with the exported model')
parser.add_argument('--line', help='The exaple text to process.')
parser.add_argument('--sample_words', help='File with one sentence per line')
parser.add_argument('--sample_tags', help='File with one sentence per line')
formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'END': '\033[0m',
}


def pretty_print(words, preds, golds=None):
    lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
    padded_words = [w.decode() + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
    padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
    print('words : {}'.format(' '.join(padded_words)))

    if golds is None:
        values = formatters
        values['TEXT'] = ' '.join(padded_preds)
        print('preds : {YELLOW}{TEXT}{END}'.format(**values))
    else:
        lengths = [max(len(w), len(p)) for w, p in zip(words, golds)]
        padded_golds = [w + (l - len(w)) * ' ' for w, l in zip(golds, lengths)]
        print('labels: {}'.format(' '.join(padded_golds)))

        inter = np.in1d(padded_preds, padded_golds)
        for idx, isinter in enumerate(inter):
            padded_preds[idx] = '{GREEN}' + padded_preds[idx]  + '{END}' if isinter else '{RED}' + padded_preds[idx]  + '{END}'
            padded_preds[idx] = padded_preds[idx].format(**formatters)
        print('preds : {}'.format(' '.join(padded_preds)))


if __name__ == '__main__':
    args = parser.parse_args()

    subdirs = [x for x in Path(args.model_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = tf.contrib.predictor.from_saved_model(latest)

    if args.line is not None:
        words = [w.encode() for w in args.line.split()]
        nwords = len(words)
        predictions = predict_fn({'words': [words], 'nwords': [nwords]})
        pretty_print(words, predictions['tags'][0])

    elif args.sample_words is not None and args.sample_tags is not None:
        with Path(args.sample_words).open('r') as f_words, \
                Path(args.sample_tags).open('r') as f_tags:
            line_words = f_words.readlines()
            line_tags = f_tags.readlines()

        idx = random.randint(0, len(line_words) - 1)
        line = line_words[idx]
        gold = line_tags[idx]

        words = [w.encode() for w in line.split()]
        nwords = len(words)
        predictions = predict_fn({'words': [words], 'nwords': [nwords]})

        golds = gold.strip().split()
        pretty_print(words, predictions['tags'][0], golds)
