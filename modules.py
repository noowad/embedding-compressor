# coding:utf-8
import tensorflow as tf
from hyperparams import Hyperparams as hp

# Import _linear
if tuple(map(int, tf.__version__.split(".")[:2])) >= (1, 6):
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell

    _linear = core_rnn_cell._linear
else:
    from tensorflow.python.ops.rnn_cell_impl import _linear


def gumbel_softmax(logits, temperature, eps=1e-10):
    '''Gumbel-Softmax'''
    U = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    # sampling from -log(-log(Uniform[0,1]))
    gumbel_distance = -tf.log(-tf.log(U + eps) + eps)
    y = logits + gumbel_distance
    return tf.nn.softmax(y / temperature)


# Encode
def encode(input_embeds):
    '''Encoder'''
    with tf.variable_scope("h"):
        h = tf.nn.tanh(_linear(input_embeds, hp.M * hp.K / 2, True))
    with tf.variable_scope("logits"):
        logits = _linear(h, hp.M * hp.K, True)
        logits = tf.log(tf.nn.softplus(logits) + 1e-8)
    logits = tf.reshape(logits, [-1, hp.M, hp.K], name="logits")
    return logits


# Decode
def decode(gumbel_output, codebooks):
    '''Decoder'''
    return tf.matmul(gumbel_output, codebooks)
