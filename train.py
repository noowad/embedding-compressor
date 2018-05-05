# coding:utf-8
from __future__ import print_function
import tensorflow as tf
import os
from tqdm import tqdm
from tensorflow.python.platform import tf_logging as logging
from hyperparams import Hyperparams as hp
import numpy as np
from modules import gumbel_softmax, encode, decode
import argparse


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
            # Discretization
            self.D = gumbel_softmax(self.logits, hp.tau_value)  # (batch_size,M,K)
            self.gumbel_output = tf.reshape(self.D, [-1, hp.M * hp.K])  # (batch_size, M * K)
            self.maxp = tf.reduce_mean(tf.reduce_max(self.D, axis=2))

            # Decoding
            self.output_embeds = decode(self.gumbel_output, self.codebooks)  # (batch_size, M*K) * (M*K, embed_size)

            # Loss
            self.loss = tf.reduce_mean(0.5 * tf.reduce_sum((self.output_embeds - self.input_embeds) ** 2, axis=1),
                                       name="loss")

            # Optimization
            self.train_vars = tf.trainable_variables()
            self.grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, self.train_vars),
                                                                  clip_norm=0.001)
            self.global_norm = tf.identity(self.global_norm, name="global_norm")
            self.optimizer = tf.train.AdamOptimizer(0.0001)
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.train_vars), name="train_op")


if __name__ == '__main__':
    if not os.path.exists('logdir'): os.makedirs('logdir')
    matrix = np.load(hp.np_embed_path)[:hp.vocab_size]
    g = Graph(matrix)
    print("Graph loaded")
    vocab_size = matrix.shape[0]
    vocab_list = list(range(vocab_size))
    # for validation
    valid_ids = np.random.RandomState(3).randint(0, vocab_size, size=(hp.batch_size * 10,)).tolist()
    with g.graph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best_valid_loss = 100000.
            for epoch in range(hp.num_epochs):
                train_loss_list = []
                train_maxp_list = []
                np.random.shuffle(vocab_list)
                train_loss = 0.
                train_maxp = 0.
                # Train
                for idx in range(0, vocab_size, hp.batch_size):
                    word_ids = vocab_list[idx:idx + hp.batch_size]
                    t_loss, t_maxp, _ = sess.run([g.loss, g.maxp, g.train_op], {g.word_ids: word_ids})
                    train_loss += t_loss
                    train_maxp += t_maxp
                train_loss /= (vocab_size / hp.batch_size)
                train_maxp /= (vocab_size / hp.batch_size)
                # Validation
                valid_loss = 0.
                valid_maxp = 0.
                for idx in range(0, len(valid_ids), hp.batch_size):
                    word_ids = valid_ids[idx:idx + hp.batch_size]
                    v_loss, v_maxp = sess.run([g.loss, g.maxp], {g.word_ids: word_ids})
                    valid_loss += v_loss
                    valid_maxp += v_maxp
                valid_loss /= (len(valid_ids) / hp.batch_size)
                valid_maxp /= (len(valid_ids) / hp.batch_size)
                print("[epoch{}] train_loss={:.2f} train_maxp={:.2f} validate_loss={:.2f} validate_maxp={:.2f}".format
                      (epoch, train_loss, train_maxp, valid_loss, valid_maxp))
                # Report
                report_token = ""
                if valid_loss <= best_valid_loss * 0.999:
                    report_token = "*"
                    best_valid_loss = valid_loss
                    saver.save(sess, 'logdir/' + hp.modelname)
                else:
                    if hp.is_earlystopping:
                        print("Early Stopping...")
                        break
    print("Done")
