"""Export model as a saved_model"""

from pathlib import Path
import json
import tensorflow as tf

# Global config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_src', './results/ned/lc-9', 'Folder where the model checkpoint is located.')
flags.DEFINE_string('params', './results/ned/params.json', 'Path to the parameters file.')
flags.DEFINE_string('model_dest', './saved_model/ned', 'Folder to save the model.')

from train import model_fn


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    receiver_tensors = {'words': words, 'nwords': nwords}
    features = {'words': words, 'nwords': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    with Path(FLAGS.params).open() as f:
        params = json.load(f)

    params['words'] = str(Path(FLAGS.data_dir, 'vocab.words.txt'))
    params['chars'] = str(Path(FLAGS.data_dir, 'vocab.chars.txt'))
    params['tags'] = str(Path(FLAGS.data_dir, 'vocab.tags.txt'))
    params['glove'] = str(Path(FLAGS.data_dir, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, FLAGS.model_src, params=params)
    estimator.export_saved_model(FLAGS.model_dest, serving_input_receiver_fn)
