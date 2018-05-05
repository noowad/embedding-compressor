# coding:utf-8
import os
import numpy as np
import tensorflow as tf
from modules import encode
from hyperparams import Hyperparams as hp
from export import Graph

if __name__ == '__main__':
    assert os.path.exists('logdir/' + hp.modelname + '.meta')
    matrix = np.load(hp.np_embed_path)[:hp.vocab_size]
    g = Graph(matrix)
    with g.graph.as_default(), tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './logdir/' + hp.modelname)

        vocab_list = list(range(matrix.shape[0]))
        distances = []
        for start_idx in range(0, hp.vocab_size, hp.batch_size):
            word_ids = vocab_list[start_idx:start_idx + hp.batch_size]
            reconstructed_vecs = sess.run(g.reconstructed_embed, {g.word_ids: word_ids})
            original_vecs = matrix[start_idx:start_idx + hp.batch_size]
            distances.extend(np.linalg.norm(reconstructed_vecs - original_vecs, axis=1).tolist())
        print np.mean(distances)
