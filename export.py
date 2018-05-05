# coding:utf-8
import os
import numpy as np
import tensorflow as tf
from modules import encode
from hyperparams import Hyperparams as hp


class Graph():
    def __init__(self, embed_matrix):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input
            self.input_matrix = tf.constant(embed_matrix, name="embed_matrix")
            self.word_ids = tf.placeholder(tf.int32, shape=[None], name="word_ids")
            # 圧縮したい分散表現　(batch_size,embed_size)
            self.input_embeds = tf.nn.embedding_lookup(self.input_matrix, self.word_ids, name="input_embeds")

            # Codebooks
            self.codebooks = tf.get_variable("codebook", [hp.M * hp.K, hp.embed_size])

            # Encoding
            self.logits = encode(self.input_embeds)  # (batch_size, M, K)
            self.codes = tf.cast(tf.argmax(self.logits, axis=2), tf.int32)  # ~ (B, M)

            # Reconstruct
            self.offset = tf.range(hp.M, dtype="int32") * hp.K
            self.codes_with_offset = self.codes + self.offset[None, :]

            self.selected_vectors = tf.gather(self.codebooks, self.codes_with_offset)  # ~ (B, M, H)
            self.reconstructed_embed = tf.reduce_sum(self.selected_vectors, axis=1)  # ~ (B, H)


if __name__ == '__main__':
    assert os.path.exists('logdir/' + hp.modelname + '.meta')
    matrix = np.load(hp.np_embed_path)[:hp.vocab_size]
    g = Graph(matrix)
    with g.graph.as_default(), tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './logdir/' + hp.modelname)

        # Dump codebook
        codebook_tensor = sess.graph.get_tensor_by_name('codebook:0')
        np.save('datas/compressed_codebook.npy', sess.run(codebook_tensor))
        # Dump codes
        with open('datas/codes.npy', 'w') as fout:
            vocab_list = list(range(matrix.shape[0]))
            for start_idx in range(0, hp.vocab_size, hp.batch_size):
                word_ids = vocab_list[start_idx:start_idx + hp.batch_size]
                codes = sess.run(g.codes, {g.word_ids: word_ids}).tolist()
                for code in codes:
                    fout.write(" ".join(map(str, code)) + "\n")
