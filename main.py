from __future__ import division
from __future__ import print_function

import argparse
from operator import itemgetter
from itertools import combinations
import time
import os

#import neptune
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn import metrics

from constants import MIN_SIDE_EFFECT_FREQUENCY
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
from utils import *
from constants import PARAMS
from adj_matrix import create_combo_adj, create_adj_matrix
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

tf.compat.v1.disable_eager_execution()

# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.compat.v1.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.compat.v1.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.compat.v1.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.compat.v1.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.compat.v1.placeholder(tf.int32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.compat.v1.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.compat.v1.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

####
# The following code uses artificially generated and very small networks.
# Expect less than excellent performance as these random networks do not have any interesting structure.
# The purpose of main.py is to show how to use the code!
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Replace dummy toy datasets used here with the actual datasets you just downloaded.
# (3) Train & test the model.
####
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for decagon')
    parser.add_argument('--no-log', default=False,
                        action='store_true',
                        help='Whether to log run or nor, default True')
    # parser.add_argument("--decagon_data_file_directory", type=str,
    #                     help="path to directory where bio-decagon-*.csv files are located, with trailing slash. "
    #                          "Default is current directory",
    #                     default='./data/input')
    # parser.add_argument("--saved_files_directory", type=str,
    #                     help="path to directory where saved files files are located, with trailing slash. "
    #                          "Default is current directory. If a decagon_model.ckpt* exists in this directory, it will "
    #                          "be loaded and evaluated, and no training will be done.",
    #                     default='./data/output')
    # parser.add_argument("--verbose", help="increase output verbosity",
    #                     action="store_true", default=True)
    args = parser.parse_args()

    if not args.no_log:
        neptune.init('Pollutants/sandbox')
    #
    # decagon_data_file_directory = args.decagon_data_file_directory
    # verbose = args.verbose
    #
    # # create pre-processed file that only has frequent side effect
    # all_combos_df = pd.read_csv(
    #     f'{decagon_data_file_directory}/bio-decagon-combo.csv')
    # side_effects_freq = all_combos_df["Polypharmacy Side Effect"].value_counts()
    # side_effects_freq = side_effects_freq[side_effects_freq >=
    #                                       MIN_SIDE_EFFECT_FREQUENCY]\
    #     .index.tolist()
    # all_combos_df = all_combos_df[
    #     all_combos_df["Polypharmacy Side Effect"].isin(side_effects_freq)]
    # all_combos_df.to_csv(
    #     f'{decagon_data_file_directory}/bio-decagon-combo-freq-only.csv',
    #     index=False)
    #
    # # use pre=processed file that only contains the most common side effects
    # drug_drug_net, combo2stitch, combo2se, se2name = load_combo_se(
    #     combo_path=(f'{decagon_data_file_directory}/bio-decagon-combo-freq-only.csv'))
    # # net is a networkx graph with genes(proteins) as nodes and protein-protein-interactions as edges
    # # node2idx maps node id to node index
    # gene_net, node2idx = load_ppi(
    #     ppi_path=(f'{decagon_data_file_directory}/bio-decagon-ppi.csv'))
    # # stitch2se maps (individual) stitch ids to a list of side effect ids
    # # se2name_mono maps side effect ids that occur in the mono file to side effect names (shorter than se2name)
    # stitch2se, se2name_mono = load_mono_se(
    #     mono_path=(f'{decagon_data_file_directory}/bio-decagon-mono.csv'))
    # # stitch2proteins maps stitch ids (drug) to protein (gene) ids
    # drug_gene_net, stitch2proteins = load_targets(
    #     targets_path=(f'{decagon_data_file_directory}/bio-decagon-targets-all.csv'))
    # # se2class maps side effect id to class name
    #
    # # this was 0.05 in the original code, but the paper says
    # # that 10% each are used for testing and validation
    # val_test_size = 0.1
    # n_genes = gene_net.number_of_nodes()
    # gene_adj = nx.adjacency_matrix(gene_net)
    # # Number of connections for each gene
    # gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
    #
    # ordered_list_of_drugs = list(drug_drug_net.nodes.keys())
    # ordered_list_of_side_effects = list(se2name.keys())
    # ordered_list_of_proteins = list(gene_net.nodes.keys())
    # ordered_list_of_se_mono = list(se2name_mono.keys())
    #
    # n_drugs = len(ordered_list_of_drugs)
    #
    # # needs to be drug vs. gene matrix (645x19081)
    # drug_gene_adj = create_adj_matrix(
    #     a_item2b_item=stitch2proteins,
    #     ordered_list_a_item=ordered_list_of_drugs,
    #     ordered_list_b_item=ordered_list_of_proteins)
    # gene_drug_adj = drug_gene_adj.transpose(copy=True)
    #
    # # TODO: Made better checkout (adjacency matrix can be partly saved from previous run
    # if not os.path.isfile("adjacency_matrices/sparse_matrix0000.npz"):
    #     drug_drug_adj_list = create_combo_adj(
    #         combo_a_item2b_item=combo2se,
    #         combo_a_item2a_item=combo2stitch,
    #         ordered_list_a_item=ordered_list_of_drugs,
    #         ordered_list_b_item=ordered_list_of_side_effects)
    #
    #     print("Saving matrices to file")
    #     # save matrices to file
    #     if not os.path.isdir("adjacency_matrices"):
    #         os.mkdir("adjacency_matrices")
    #     for i in range(len(drug_drug_adj_list)):
    #         sp.save_npz('adjacency_matrices/sparse_matrix%04d.npz' % (i,),
    #                     drug_drug_adj_list[i].tocoo())
    # else:
    #     drug_drug_adj_list = []
    #     print("Loading adjacency matrices from file.")
    #     for i in range(len(ordered_list_of_side_effects)):
    #         drug_drug_adj_list.append(
    #             sp.load_npz('adjacency_matrices' +
    #                         f'/sparse_matrix%04d.npz' % i).tocsr())
    # # Number of connections for each drug
    # drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj
    #                      in drug_drug_adj_list]
    #
    #
    #
    # adj_mats_orig = {
    #     (0, 0): [gene_adj],
    #     (0, 1): [gene_drug_adj],
    #     (1, 0): [drug_gene_adj],
    #     (1, 1): drug_drug_adj_list,
    # }
    # degrees = {
    #     0: [gene_degrees],
    #     1: drug_degrees_list,
    # }
    #
    # # featureless (genes)
    # gene_feat = sp.identity(n_genes)
    # gene_nonzero_feat, gene_num_feat = gene_feat.shape
    # gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())
    #
    # # features (drugs)
    # se_mono2idx = dict(zip(ordered_list_of_se_mono,
    #                        range(len(ordered_list_of_se_mono))))
    # # Create sparse matrix with rows -- genes features.
    # # Gene feature -- binary vector with length = num of mono se.
    # # feature[i] = 1 <=> gene has ith mono se
    # drug_feat = create_adj_matrix(
    #     a_item2b_item=stitch2se,
    #     ordered_list_a_item=ordered_list_of_drugs,
    #     ordered_list_b_item=ordered_list_of_se_mono)
    # drug_nonzero_feat, drug_num_feat = drug_feat.shape
    # drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

##############
    val_test_size = 0.05
    n_genes = 500
    n_drugs = 400
    n_drugdrug_rel_types = 3
    gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)

    gene_adj = nx.adjacency_matrix(gene_net)
    gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

    gene_drug_adj = sp.csr_matrix(
        (10 * np.random.randn(n_genes, n_drugs) > 15).astype(int))
    drug_gene_adj = gene_drug_adj.transpose(copy=True)

    drug_drug_adj_list = []
    tmp = np.dot(drug_gene_adj, gene_drug_adj)
    for i in range(n_drugdrug_rel_types):
        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in combinations(list(range(n_drugs)), 2):
            if tmp[d1, d2] == i + 4:
                mat[d1, d2] = mat[d2, d1] = 1.
        drug_drug_adj_list.append(sp.csr_matrix(mat))
    drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj
                         in drug_drug_adj_list]

    # data representation
    adj_mats_orig = {
        (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
        (0, 1): [gene_drug_adj],
        (1, 0): [drug_gene_adj],
        (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in
                                      drug_drug_adj_list],
    }
    degrees = {
        0: [gene_degrees, gene_degrees],
        1: drug_degrees_list + drug_degrees_list,
    }

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

    # features (drugs)
    drug_feat = sp.identity(n_drugs)
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

####################
    # data representation
    num_feat = {
        0: gene_num_feat,
        1: drug_num_feat,
    }
    nonzero_feat = {
        0: gene_nonzero_feat,
        1: drug_nonzero_feat,
    }
    feat = {
        0: gene_feat,
        1: drug_feat,
    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
    edge_type2decoder = {
        (0, 0): 'bilinear',
        (0, 1): 'bilinear',
        (1, 0): 'bilinear',
        (1, 1): 'dedicom',
    }

    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)

    ###########################################################
    #
    # Settings and placeholders
    #
    ###########################################################

    # flags = tf.compat.v1.app.flags
    # FLAGS = flags.FLAGS
    # flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
    # flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    # flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
    # flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
    # flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    # flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    # flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
    # flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
    # flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
    # flags.DEFINE_boolean('bias', True, 'Bias term.')


    if not args.no_log:
        neptune.create_experiment(name='example_with_parameters',
                                  params=PARAMS,
                                  upload_stdout=True,
                                  upload_stderr=True,
                                  send_hardware_metrics=True,
                                  upload_source_files='**/*.py')
        neptune.set_property("val_test_size", val_test_size)
    # Important -- Do not evaluate/print validation performance every iteration as it can take
    # substantial amount of time
    PRINT_PROGRESS_EVERY = 150

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)

    ###########################################################
    #
    # Create minibatch iterator, model and optimizer
    #
    ###########################################################

    print("Create minibatch iterator")
    path_to_split = f'data/split/{val_test_size}'
    need_sample_edges = not (os.path.isdir(path_to_split) and
                             len(os.listdir(path_to_split)) == 6)
    minibatch = EdgeMinibatchIterator(
        adj_mats=adj_mats_orig,
        feat=feat,
        edge_types=edge_types,
        batch_size=PARAMS['batch_size'],
        val_test_size=val_test_size,
        path_to_split=path_to_split,
        need_sample_edges=need_sample_edges
    )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    with tf.compat.v1.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=PARAMS['batch_size'],
            margin=PARAMS['max_margin']
        )

    print("Initialize session")
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    feed_dict = {}

    ###########################################################
    #
    # Train model
    #
    ###########################################################

    print("Train model")
    for epoch in range(PARAMS['epochs']):

        minibatch.shuffle()
        itr = 0
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
            feed_dict = minibatch.update_feed_dict(
                feed_dict=feed_dict,
                dropout=PARAMS['dropout'],
                placeholders=placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if itr % PRINT_PROGRESS_EVERY == 0:
                val_auc, val_auprc, val_apk = get_accuracy_scores(
                    minibatch.val_edges, minibatch.val_edges_false,
                    minibatch.idx2edge_type[minibatch.current_edge_type_idx])

                print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))
                if not args.no_log:
                    neptune.log_metric("val_roc", val_auc, timestamp=time.time())
                    neptune.log_metric("val_apk", val_apk, timestamp=time.time())
                    neptune.log_metric("val_auprc", val_auprc,
                                       timestamp=time.time())
                    neptune.log_metric("train_loss", train_cost,
                                       timestamp=time.time())

            itr += 1

    print("Optimization finished!")

    for et in range(num_edge_types):
        roc_score, auprc_score, apk_score = get_accuracy_scores(
            minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
        print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
        print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
        print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
        print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
        print()
        if not args.no_log:
            neptune.log_metric("ROC-AUC", roc_score)
            neptune.log_metric("AUPRC", auprc_score)
            neptune.log_metric("AP@k score", apk_score)
    if not args.no_log:
        neptune.stop()
