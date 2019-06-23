from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from supervised_model import SupervisedGraphsage, TwoChannelGraphsage, SeqGraphsage, SeqTestGraphsage
from minibatch import NodeMinibatchIterator, SeqNodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler, SeqUniformNeighborSampler
from graphsage import SAGEInfo
from utils import load_data, load_seq_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False, 'log')


flags.DEFINE_string('model', 'mean', 'model version')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('model_size', 'small', 'small or big')
flags.DEFINE_string('train_prefix', '', 'train path')

flags.DEFINE_boolean('sequential', True, 'sequential dataset or not')
flags.DEFINE_integer('epochs', 10, 'epochs')
flags.DEFINE_float('dropout', 0.5, 'dropout')
flags.DEFINE_float('weight_decay', 0.02, 'weight decay for l2-norm')
flags.DEFINE_float('weight_value', 1.2, 'weight value for positive classes')
flags.DEFINE_integer('max_degree', 32, 'max degree')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0,  'number of samples in layer 3')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim')
flags.DEFINE_integer('dim_2', 128, 'size of output dim')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('identity_dim', 100, 'identity dimension')
flags.DEFINE_boolean('split_class', True, 'split category into old and new')
flags.DEFINE_integer('num_classes', 123, 'item numbers')
flags.DEFINE_integer('max_input_classes', 5, 'output length')
flags.DEFINE_boolean('two_channel', True, 'normal and abnormal')
flags.DEFINE_integer('num_steps', 5, 'step numbers')
flags.DEFINE_string('base_log_dir', './log/', 'log save path')
flags.DEFINE_integer('validate_iter', 20, 'how often to run validation')
flags.DEFINE_integer('validate_batch_size', 128, 'batch size in valiation')
flags.DEFINE_integer('print_every', 20, 'how often to print')
flags.DEFINE_integer('gpu', 1, 'number of gpu')
flags.DEFINE_boolean('random_context', True, 'use random context')
flags.DEFINE_integer('max_total_steps', 10**5, 'max steps')
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

CPU_MEM_FRACTION = 0.8
print(FLAGS.batch_size)
print(FLAGS.two_channel)

def construct_placeholders():    
    if not FLAGS.sequential:
        print("using single channel placeholders")
        placeholders = {
            'labels' : tf.placeholder(tf.float32, shape=(None, FLAGS.num_steps, FLAGS.num_classes), name='labels'),
            'batch'  : tf.placeholder(tf.int32, shape=(None), name='batch'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'weights': tf.placeholder(tf.float32, shape=(None, FLAGS.num_steps, FLAGS.num_classes), name='weights'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        }
    else:
        print("using two channel placeholders")
        placeholders = {
            'labels_rec': tf.placeholder(tf.float32, shape=(None, FLAGS.num_steps, FLAGS.num_classes), name='labels_rec'),
            'labels_abn': tf.placeholder(tf.float32, shape=(None, FLAGS.num_steps, FLAGS.num_classes), name='labels_abn'),
            'weights_rec': tf.placeholder(tf.float32, shape=(None, FLAGS.num_steps, FLAGS.num_classes), name='weights_rec'),
            'weights_abn': tf.placeholder(tf.float32, shape=(None, FLAGS.num_steps, FLAGS.num_classes), name='weights_abn'),
            'batch': tf.placeholder(tf.int32, shape=(None), name='batch'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        }
    return placeholders


def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true.round(), y_pred, average="micro"), metrics.f1_score(y_true.round(), y_pred, average="macro")

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
                model=FLAGS.model,
                model_size=FLAGS.model_size,
                lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.outputs, model.loss], feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time()- t_test)


def incremental_evaluate(sess, model, minibatch_iter, class_map, test_labels, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    while not finished:
        feed_dict_val, batch_labels, finished, batch_nodes = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.outputs, model.loss], feed_dict=feed_dict_val)
        preds = node_outs_val[0]
        val_losses.append(node_outs_val[1])
        for idx, node in enumerate(batch_nodes):
            for cate in test_labels[-1][node]["pos"]:
                if cate in class_map:
                    labels.append(1)
                    val_preds.append(int(preds[idx,class_map[cate]] > 0.5))
            for cate in test_labels[-1][node]["neg"]:
                if cate in class_map:
                    labels.append(0)
                    val_preds.append(int(preds[idx, class_map[cate]] > 0.5))
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)

def incremental_evaluate_with_split_class(sess, model, minibatch_iter, class_map, test_labels, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    labels_old = []
    preds_old  = []
    labels_new = []
    preds_new  = []
    iter_num = 0
    while not finished: 
        feed_dict_val, batch_labels, finished, batch_nodes = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        if not FLAGS.two_channel:
            node_outs_val = sess.run([model.outputs, model.loss], feed_dict = feed_dict_val)
            preds = node_outs_val[0]
            val_losses.append(node_outs_val[1])
        else:
            output_old, output_new, output_loss = sess.run([model.output_rec, model.output_abn, model.loss], feed_dict = feed_dict_val)
            val_losses.append(output_loss)
        for idx, node in enumerate(batch_nodes):
            for cate in test_labels[-1][node]["old"]:
                if cate in class_map:
                    labels_old.append(1)
                    if not FLAGS.two_channel:
                        preds_old.append(int(preds[idx, class_map[cate]] > 0.5))
                    else:
                        preds_old.append(int(output_old[idx, class_map[cate]] > 0.5))
            for cate in test_labels[-1][node]["neg"][:len(test_labels[-1][node]["old"])]:
                if cate in class_map:
                    labels_old.append(0)
                    if not FLAGS.two_channel:
                        preds_old.append(int(preds[idx, class_map[cate]] > 0.5))
                    else:
                        preds_old.append(int(output_old[idx, class_map[cate]] > 0.5))
            for cate in test_labels[-1][node]["new"]:
                if cate in class_map:
                    labels_new.append(1)
                    if not FLAGS.two_channel:
                        preds_new.append(int(preds[idx, class_map[cate]] > 0.5))
                    else:
                        preds_new.append(int(output_new[idx, class_map[cate]] > 0.5))
            for cate in test_labels[-1][node]["neg"][:len(test_labels[-1][node]["new"])]:
                if cate in class_map:
                    labels_new.append(0)
                    if not FLAGS.two_channel:
                        preds_new.append(int(preds[idx, class_map[cate]] > 0.5))
                    else:
                        preds_new.append(int(output_new[idx, class_map[cate]] > 0.5))
        iter_num += 1
    preds_old = np.vstack(preds_old)
    labels_old = np.vstack(labels_old)
    preds_new = np.vstack(preds_new)
    labels_new = np.vstack(labels_new)
    f1_old = calc_f1(labels_old, preds_old)
    f1_new = calc_f1(labels_new, preds_new)
    return np.mean(val_losses), f1_old[0], f1_old[1], f1_new[0], f1_new[1], (time.time() - t_test)


def train(train_data, test_data=None):
    
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    all_class = train_data[4]
    class_map = train_data[5]
    test_class = train_data[6]

    num_classes = FLAGS.num_classes

    if not features is None:
        features = np.vstack([features, np.zeros((features.shape[1], ))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()

    minibatch = SeqNodeMinibatchIterator(G,
                    id_map,
                    placeholders,
                    all_class,
                    num_classes,
                    batch_size=FLAGS.batch_size,
                    max_degree=FLAGS.max_degree,
                    context_pairs = context_pairs,
                    num_steps = FLAGS.num_steps)
    adj_info_ph = tf.placeholder(tf.int32, shape=np.array(minibatch.adj).shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    
    sampler = SeqUniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
    if FLAGS.samples_2 != 0:
        layer_infos.append(SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2))
    if FLAGS.samples_3 != 0:
        layer_infos.append(SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2))
    if FLAGS.sequential:
        print("using sequential two channel inputs")
        model = SeqGraphsage(num_classes, placeholders,
                            features,
                            adj_info,
                            minibatch.deg,
                            layer_infos,
                            model_size=FLAGS.model_size,
                            identity_dim=FLAGS.identity_dim,
                            num_steps = FLAGS.num_steps,
                            logging=True)
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: np.array(minibatch.adj)})

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, np.array(minibatch.adj))
    val_adj_info   = tf.assign(adj_info, np.array(minibatch.test_adj))
    best_vani_f1_mic , best_vani_f1_mac,  best_burst_f1_mic, best_burst_f1_mac = 0, 0 , 0, 0
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()
        iter = 0
        print("Epoch: %04d" % (epoch+1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            t = time.time()
            outs = sess.run([merged, model.opt_op, model.loss, model.outputs], feed_dict=feed_dict)
            train_cost = outs[2]
            if iter % FLAGS.validate_iter == 0:
                
                sess.run(val_adj_info.op)
                if not FLAGS.split_class:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, class_map, test_class, FLAGS.validate_batch_size)
                else:
                    val_cost, old_f1_mic, old_f1_mac, new_f1_mic, new_f1_mac, duration = incremental_evaluate_with_split_class(sess, model, minibatch, class_map, test_class, FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                if not FLAGS.split_class:
                    print("Iter: {:04d}\ntrain_loss= {:.5f} test_loss= {:.5f}\nval_f1_mic= {:.5f} val_f1_mac= {:.5f} time= {:.5f}".format(iter, train_cost, val_cost, val_f1_mic, val_f1_mac, avg_time))
                else:
                    print("Iter: {:04d}\ntrain_loss= {:.5f} test_loss= {:.5f}\nvanilla_f1_mic= {:.5f} vanilla_f1_mac= {:.5f} burst_f1_mic= {:.5f} burst_f1_mac= {:.5f} time= {:.5f}".format(iter, train_cost, val_cost, old_f1_mic, old_f1_mac, new_f1_mic, new_f1_mac, avg_time))
                    if old_f1_mic > best_vani_f1_mic:
                        best_vani_f1_mic = old_f1_mic
                        best_vani_f1_mac = old_f1_mac
                        best_burst_f1_mic = new_f1_mic
                        best_burst_f1_mac = new_f1_mac
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break
        if total_steps > FLAGS.max_total_steps:
            break
    
    print("Optimization finished")
    if FLAGS.split_class:
        print("Results:\nvanilla_f1_micro= {:.5f} vanilla_f1_macro={:.5f} burst_f1_micro= {:.5f} burst_f1_macro= {:.5f}".format( best_vani_f1_mic, best_vani_f1_mac, best_burst_f1_mic, best_burst_f1_mac))

def main(argv=None):
    print("Loading training data ..")
    if FLAGS.sequential:
        train_data = load_seq_data(FLAGS.train_prefix, num_steps=FLAGS.num_steps, split_class=FLAGS.split_class)
    else:
        train_data = load_data(FLAGS.train_prefix, split_class = FLAGS.split_class)
    print("Done loading training data ..")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()
