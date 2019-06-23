from __future__ import division
from __future__ import print_function

from layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists



class SeqUniformNeighborSampler(Layer):

    def __init__(self, adj_info, **kwargs):
        super(SeqUniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = tf.unstack(adj_info, axis=0)
        print("adj_info :", tf.shape(self.adj_info))

    def _call(self, inputs):
        ids, num_samples, aid = inputs
        print(aid)
        adj_lists = tf.nn.embedding_lookup(self.adj_info[aid], ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
