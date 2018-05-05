from __future__ import absolute_import, division, print_function
import numpy as np
from hyperparams import Hyperparams as hp
from argparse import ArgumentParser


def make_glove2numpy(glove_path='./datas/glove.6B.100d.txt', dim=100):
    embed_path = glove_path.replace(".txt", ".npy")
    word_path = glove_path.replace(".txt", ".word")
    print("convert {} to {}".format(glove_path, embed_path))
    with open(word_path, "w") as f_wordout:
        lines = open(glove_path, 'r').read().splitlines()
        embed_matrix = np.zeros((len(lines), dim), dtype='float32')
        for i, line in enumerate(lines):
            parts = line.split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype='float32')
            embed_matrix[i] = vec
            f_wordout.write(word + "\n")
    np.save(embed_path, embed_matrix)
