"""Interact with a model"""

from pathlib import Path
import functools
import itertools
import json

import numpy as np
import tensorflow as tf

from main import generator_fn

LINE = 'Jose trabaja para CNN en Sao Paulo , Brasil .'
DATADIR = '../dataset/conll2002/esp'
PARAMS = './results/params.json'
MODELDIR = './results/esp/run-2'


def input_fn(words, tags):
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags, sort_by_scores=True, sample_size=5),
        output_shapes=shapes, output_types=types).repeat(2)

    return dataset


if __name__ == '__main__':
    dataset = input_fn(DATADIR + '/train.words.txt', DATADIR + '/train.tags.txt')
    iterator = dataset.make_one_shot_iterator()
    node = iterator.get_next()
    with tf.Session() as sess:
        for i in range(10):
            print(sess.run(node))

