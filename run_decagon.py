import os
import numpy as np
import scipy.sparse as sp
from decagon.utility import rank_metrics, preprocessing
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
import tensorflow as tf
from typing import Dict
import time
from sklearn import metrics
from operator import itemgetter
import  neptune

tf.compat.v1.disable_eager_execution()


class RunDecagon:
    """
    Attributes
    ----------
    adj_mats: Dict[Tuple[int, int], List[sp.csr_matrix]]
    degrees: Dict[int, List[int]]
    edge_type2dim: Dict[Tuple[int, int], List[int]
    edge_type2decoder: Dict[Tuple[int, int], str]
    edge_types: Dict[Tuple[int, int], int]
    num_edge_types: int
    num_feat: Dict[int, int]
    nonzero_feat: Dict[int, int]
    feat: Dict[int, sp.csr_matrix]

    minibatch: EdgeMinibatchIterator
    placeholders: Dict[str, tf.compat.v1.placeholder]
    model: DecagonModel
    opt: DecagonOptimizer
    """

    def __init__(self):
        pass

    def _adjacency(self, adj_path: str) -> None:
        """
        Create self.adj_mats, self.degrees
        Parameters
        ----------
        adj_path: str
            path for saving/loading adjacency matrices

        Returns
        -------

        Notes
        -----
        self.adj_mats: Dict[Tuple[int, int], List[sp.csr_matrix]]
            from edge type to list of adjacency matrices for each edge class
            (e.g. (1, 1): list of drug-drug adjacency matrices for each se class)
            In our case all matrix in adj_mats are symmetric
        self.degrees: Dict[int, List[int]]
            number of connections for each node (0: genes, 1: drugs)

        """
        raise NotImplementedError()

    def _nodes_features(self) -> None:
        """
        Create self.num_feat, self.nonzero_feat, self.feat.

        Returns
        -------

        Notes
        -----
        One-hot encoding as genes features.
        Binary vectors with presence of different side effects as drugs features
        self.num_feat: Dict[int, int]
            number of elements in feature vector for 0: -genes, for 1: -drugs.
        self.nonzero_feat: Dict[int, int]
            number of all features for 0: -gene and 1: -drug nodes.
            All features should be nonzero! ????????????
            TODO: What to do with zero features??
            e.g., it is in format 0: num of genes in graph, 1: num of drugs.
        self.feat: Dict[int, sp.csr_matrix]
            from edge type (0 = gene, 1 = drug) to feature matrix.
            row in feature matrix = embedding of one node.
        """
        raise NotImplementedError()

    def _edge_types_info(self) -> None:
        """
        Create self.edge_type2dim, self.edge_type2decoder, self.edge_types,
        self.num_edge_types
        Returns
        -------

        Notes
        -----
        self.edge_type2dim: Dict[Tuple[int, int], List[int]
            from edge type to list of shapes all its adjacency matrices.
        self.edge_type2decoder: Dict[Tuple[int, int], str]
            from edge type to decoder type
            (we use different decompositions for different edges types)
        self.edge_types: Dict[Tuple[int, int], int]
            from edge type to number of classes of these edge type
            (e. g. (1, 1): number of se)
        self.num_edge_types: int
            number of all edge types (considering all classes)

        """
        self.edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in
                              self.adj_mats.items()}
        self.edge_type2decoder = {
            (0, 0): 'bilinear',
            (0, 1): 'bilinear',
            (1, 0): 'bilinear',
            (1, 1): 'dedicom',
        }

        self.edge_types = {k: len(v) for k, v in self.adj_mats.items()}
        self.num_edge_types = sum(self.edge_types.values())
        print(f'Edge types {self.num_edge_types}')

    def _minibatch_iterator_init(self, path_to_split: str, batch_size: int,
                                 val_test_size: float) -> None:
        """
        Create minibatch iterator (self.minibatch)
        Parameters
        ----------
        path_to_split: str
            path to save train, test and validate edges.
            If it consist needed edges, they will be loaded.
            Else they will be calculated and saved.
        batch_size: int
            Minibatch size.
        val_test_size: float
            proportion to split edges into train, test and validate.

        Returns
        -------

        """
        print('Create minibatch iterator')
        need_sample_edges = not (os.path.isdir(path_to_split) and
                                 len(os.listdir(path_to_split)) == 6)
        self.minibatch = EdgeMinibatchIterator(
            adj_mats=self.adj_mats,
            feat=self.feat,
            edge_types=self.edge_types,
            batch_size=batch_size,
            val_test_size=val_test_size,
            path_to_split=path_to_split,
            need_sample_edges=need_sample_edges
        )

    def _construct_placeholders(self) -> None:
        """
        Create self.placeholders.

        Returns
        -------

        Notes
        _____
        Placeholders - input data in tf1.
        """
        print("Defining placeholders")
        self.placeholders = {
            'batch': tf.compat.v1.placeholder(tf.int32, name='batch'),
            'batch_edge_type_idx':
                tf.compat.v1.placeholder(tf.int32, shape=(),
                                         name='batch_edge_type_idx'),
            'batch_row_edge_type':
                tf.compat.v1.placeholder(tf.int32, shape=(),
                                         name='batch_row_edge_type'),
            'batch_col_edge_type':
                tf.compat.v1.placeholder(tf.int32, shape=(),
                                         name='batch_col_edge_type'),
            'degrees': tf.compat.v1.placeholder(tf.int32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        }

        adj_placeholders = {'adj_mats_%d,%d,%d' % (i, j, k):
                                tf.compat.v1.sparse_placeholder(tf.float32)
                            for i, j in self.edge_types
                            for k in range(self.edge_types[i, j])}
        self.placeholders.update(adj_placeholders)

        features_placeholders = {'feat_%d' % i:
                                     tf.compat.v1.sparse_placeholder(tf.float32)
                                 for i, _ in self.edge_types}
        self.placeholders.update(features_placeholders)

    def _model_init(self) -> None:
        """
        Create self.model
        Returns
        -------

        """
        print("Create model")
        self.model = DecagonModel(
            placeholders=self.placeholders,
            num_feat=self.num_feat,
            nonzero_feat=self.nonzero_feat,
            edge_types=self.edge_types,
            decoders=self.edge_type2decoder,
        )

    def _optimizer_init(self, batch_size: int, max_margin: float) -> None:
        """
        Create self.opt.
        Parameters
        ----------
        batch_size: int
            Minibatch size.
        max_margin: float
            Max margin parameter in hinge loss.

        Returns
        -------

        """
        print("Create optimizer")
        with tf.compat.v1.name_scope('optimizer'):
            self.opt = DecagonOptimizer(
                embeddings=self.model.embeddings,
                latent_inters=self.model.latent_inters,
                latent_varies=self.model.latent_varies,
                degrees=self.degrees,
                edge_types=self.edge_types,
                edge_type2dim=self.edge_type2dim,
                placeholders=self.placeholders,
                batch_size=batch_size,
                margin=max_margin
            )

    def _get_accuracy_scores(self, sess, edges_pos, edges_neg, edge_type):
        # TODO: разобрать
        self.feed_dict.update({self.placeholders['dropout']: 0})
        self.feed_dict.update({self.placeholders['batch_edge_type_idx']:
                              self.minibatch.edge_type2idx[edge_type]})
        self.feed_dict.update({self.placeholders['batch_row_edge_type']: edge_type[0]})
        self.feed_dict.update({self.placeholders['batch_col_edge_type']: edge_type[1]})
        rec = sess.run(self.opt.predictions, feed_dict=self.feed_dict)

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
            assert self.adj_mats[edge_type[:2]][edge_type[2]][
                       u, v] == 1, 'Problem 1'

            actual.append(edge_ind)
            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_neg = []
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds_neg.append(score)
            assert self.adj_mats[edge_type[:2]][edge_type[2]][
                       u, v] == 0, 'Problem 0'

            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_all = np.hstack([preds, preds_neg])
        preds_all = np.nan_to_num(preds_all)
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        predicted = \
        list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

        roc_sc = metrics.roc_auc_score(labels_all, preds_all)
        aupr_sc = metrics.average_precision_score(labels_all, preds_all)
        apk_sc = rank_metrics.apk(actual, predicted, k=50)

        return roc_sc, aupr_sc, apk_sc

    def _run_epoch(self, sess: tf.compat.v1.Session, dropout: float,
                   print_progress_every: int, epoch: int, no_log: bool) -> None:
        """
        Run one epoch
        Parameters
        ----------
        sess: tf.compat.v1.Session
            initialize tf session
        dropout: float
            Dropout rate (1 - keep probability).
        print_progress_every: int
            Print statistic every print_progress_every iterations.
        epoch: int
            Number of current epoch (for printing statistic).

        Returns
        -------

        """
        self.minibatch.shuffle()
        itr = 0
        while not self.minibatch.end():
            # Construct feed dictionary
            self.feed_dict = self.minibatch.next_minibatch_feed_dict(
                placeholders=self.placeholders)
            self.feed_dict = self.minibatch.update_feed_dict(
                feed_dict=self.feed_dict,
                dropout=dropout,
                placeholders=self.placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([self.opt.opt_op, self.opt.cost,
                             self.opt.batch_edge_type_idx], feed_dict=self.feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if itr % print_progress_every == 0:
                val_auc, val_auprc, val_apk = self._get_accuracy_scores(
                    sess, self.minibatch.val_edges, self.minibatch.val_edges_false,
                    self.minibatch.idx2edge_type[
                        self.minibatch.current_edge_type_idx])

                print("Epoch:", "%04d" % (epoch + 1), "Iter:",
                      "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "val_roc=", "{:.5f}".format(val_auc), "val_auprc=",
                      "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk), "time=",
                      "{:.5f}".format(time.time() - t))
                if not no_log:
                    neptune.log_metric("val_roc", val_auc, timestamp=time.time())
                    neptune.log_metric("val_apk", val_apk, timestamp=time.time())
                    neptune.log_metric("val_auprc", val_auprc,
                                       timestamp=time.time())
                    neptune.log_metric("train_loss", train_cost,
                                       timestamp=time.time())
            itr += 1

    def run(self, adj_path:str, path_to_split: str, val_test_size: float,
            batch_size: int, num_epochs: int, dropout:float, max_margin: float,
            print_progress_every: int, no_log=True):
        """
        Run Decagon.
        Parameters
        ----------
        adj_path: str
            path for saving/loading adjacency matrices
        path_to_split: str
            path to save train, test and validate edges.
            If it consist needed edges, they will be loaded.
            Else they will be calculated and saved.
        batch_size: int
            Minibatch size.
        val_test_size: float
            proportion to split edges into train, test and validate.
        num_epochs: int
            number of training epochs
        dropout: float
            Dropout rate (1 - keep probability).
        print_progress_every: int
            Print statistic every print_progress_every iterations
        max_margin

        Returns
        -------

        """
        self._adjacency(adj_path)
        self._nodes_features()
        self._edge_types_info()
        self._construct_placeholders()
        self._minibatch_iterator_init(path_to_split, batch_size, val_test_size)
        self._model_init()
        self._optimizer_init(batch_size, max_margin)
        print("Initialize session")
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        self.feed_dict = {}
        for epoch in range(num_epochs):
            self._run_epoch(sess, dropout, print_progress_every, epoch, no_log)
        print("Optimization finished!")

        for et in range(self.num_edge_types):
            roc_score, auprc_score, apk_score = self._get_accuracy_scores(
                sess, self.minibatch.test_edges, self.minibatch.test_edges_false,
                self.minibatch.idx2edge_type[et])
            print("Edge type=",
                  "[%02d, %02d, %02d]" % self.minibatch.idx2edge_type[et])
            print("Edge type:", "%04d" % et, "Test AUROC score",
                  "{:.5f}".format(roc_score))
            print("Edge type:", "%04d" % et, "Test AUPRC score",
                  "{:.5f}".format(auprc_score))
            print("Edge type:", "%04d" % et, "Test AP@k score",
                  "{:.5f}".format(apk_score))
            print()
        if not no_log:
            neptune.log_metric("ROC-AUC", roc_score)
            neptune.log_metric("AUPRC", auprc_score)
            neptune.log_metric("AP@k score", apk_score)



