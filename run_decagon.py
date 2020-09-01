import os
from abc import ABCMeta

import numpy as np
import scipy.sparse as sp

from constants import MODEL_SAVE_PATH, MODEL_TO_UPLOAD
from decagon.utility import rank_metrics
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
import tensorflow as tf
from typing import Dict, Tuple, List, NoReturn
import time
from scipy.special import expit
from sklearn import metrics
from datetime import datetime

tf.compat.v1.disable_eager_execution()


class RunDecagon(metaclass=ABCMeta):
    """
    Abstract class of Decagon runner.
    Different subclasses define specific behavior
    (e.g. run on synthetic data or real).


    Attributes
    ----------
    adj_mats : Dict[Tuple[int, int], List[sp.csr_matrix]]
        From edge type to list of adjacency matrices for each edge class
        (e.g. (1, 1): list of drug-drug adjacency matrices for each se class).
        In our case all matrix in adj_mats are symmetric.
    degrees : Dict[int, List[int]]
        Number of connections for each node (0: genes, 1: drugs).

    edge_type2dim : Dict[Tuple[int, int], List[int]
        From edge type to list of shapes all its adjacency matrices.
    edge_type2decoder : Dict[Tuple[int, int], str]
        From edge type to decoder type
        (we use different decompositions for different edges types).
    edge_types : Dict[Tuple[int, int], int]
        From edge type to number of classes of these edge type
        (e. g. (1, 1): number of se).
    num_edge_types : int
        Number of all edge types (considering all classes).
    symmetry_types_groups : List[List]
        Should contains lists with len in {1, 2}.
        All types of edges splits into groups of symmetry.
        E. g. symmetry_types_groups = [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]].
        Two types from one group of symmetry have same edges, differing only in direction
        (e.g (0, 1) has protein -> drug edges and (1, 0) has drug -> protein edges).

    num_feat : Dict[int, int]
        Number of elements in feature vector for 0: -genes, for 1: -drugs.
    nonzero_feat : Dict[int, int]
        Number of all features for 0: -gene and 1: -drug nodes.
    feat : Dict[int, sp.csr_matrix]
        From edge type (0 = gene, 1 = drug) to feature matrix.
        Row in feature matrix = embedding of one node.

    minibatch : EdgeMinibatchIterator
        Minibatch iterator.
    placeholders : Dict[str, tf.compat.v1.placeholder]
        Variables for input data in decagon model.
    model : DecagonModel
        Decagon model (encoder + decoder).
    opt : DecagonOptimizer
        Optimizer of decagon weigts.
    """

    def __init__(self):
        self.adj_mats = None
        self.degrees = None
        self.num_feat = None
        self.nonzero_feat = None
        self.feat = None
        self.edge_type2dim = None
        self.edge_type2decoder = None
        self.edge_types = None
        self.num_edge_types = None

        self.minibatch = None
        self.opt = None
        self.placeholders = None
        self.model = None
        self.feed_dict = None

        pass

    def _adjacency(self, adj_path: str) -> NoReturn:
        """
        Create self.adj_mats, self.degrees.

        Parameters
        ----------
        adj_path : str
            path for saving/loading adjacency matrices.

        Notes
        -----
        self.adj_mats: Dict[Tuple[int, int], List[sp.csr_matrix]]
            From edge type to list of adjacency matrices for each edge class
            (e.g. (1, 1): list of drug-drug adjacency matrices for each se class)
            In our case all matrix in adj_mats are symmetric
        self.degrees: Dict[int, List[int]]
            Number of connections for each node (0: genes, 1: drugs)

        """
        raise NotImplementedError()

    def _nodes_features(self) -> NoReturn:
        """
        Create self.num_feat, self.nonzero_feat, self.feat.

        Returns
        -------

        Notes
        -----
        self.num_feat : Dict[int, int]
            Number of elements in feature vector for 0: -genes, for 1: -drugs.
        self.nonzero_feat : Dict[int, int]
            Number of all features for 0: -gene and 1: -drug nodes.
            All features should be nonzero!??
            TODO: What to do with zero features??
            E.g., it is in format 0: num of genes in graph, 1: num of drugs.
        self.feat : Dict[int, sp.csr_matrix]
            From edge type (0 = gene, 1 = drug) to feature matrix.
            Row in feature matrix = embedding of one node.

        """
        raise NotImplementedError()

    def _edge_types_info(self) -> NoReturn:
        """
        Create self.edge_type2dim, self.edge_type2decoder, self.edge_types,
        self.num_edge_types.

        Notes
        -----
        self.edge_type2dim : Dict[Tuple[int, int], List[int]
            From edge type to list of shapes all its adjacency matrices.
        self.edge_type2decoder : Dict[Tuple[int, int], str]
            From edge type to decoder type
            (we use different decompositions for different edges types).
        self.edge_types : Dict[Tuple[int, int], int]
            From edge type to number of classes of these edge type
            (e. g. (1, 1): number of se).
        self.num_edge_types : int
            Number of all edge types (considering all classes).

        """
        self.edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in
                              self.adj_mats.items()}
        self.edge_type2decoder = {
            (0, 0): 'bilinear',
            (0, 1): 'bilinear',
            (1, 0): 'bilinear',
            (1, 1): 'dedicom',
        }
        self.symmetry_types_groups = [
            [(0, 0)],
            [(0, 1), (1, 0)],
            [(1, 1)]
        ]

        self.edge_types = {k: len(v) for k, v in self.adj_mats.items()}
        self.num_edge_types = sum(self.edge_types.values())
        print(f'Edge types {self.num_edge_types}')

    def _minibatch_iterator_init(self, path_to_split: str, batch_size: int,
                                 val_test_size: float) -> NoReturn:
        """
        Create minibatch iterator (self.minibatch).

        Parameters
        ----------
        path_to_split : str
            Path to save train, test and validate edges.
            If it consist needed edges, they will be loaded.
            Else they will be calculated and saved.
        batch_size : int
            Minibatch size.
        val_test_size : float
            Proportion to split edges into train, test and validate.

        """
        print('Create minibatch iterator')
        need_sample_edges = not (os.path.isdir(path_to_split) and
                                 len(os.listdir(path_to_split)) == 6)
        self.minibatch = EdgeMinibatchIterator(
            adj_mats=self.adj_mats,
            feat=self.feat,
            edge_types=self.edge_types,
            symmetry_types_groups=self.symmetry_types_groups,
            batch_size=batch_size,
            val_test_size=val_test_size,
            path_to_split=path_to_split,
            need_sample_edges=need_sample_edges
        )

    def _construct_placeholders(self) -> NoReturn:
        """
        Create self.placeholders.

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

    def _model_init(self) -> NoReturn:
        """
        Create self.model.

        """
        print("Create model")
        self.model = DecagonModel(
            placeholders=self.placeholders,
            num_feat=self.num_feat,
            nonzero_feat=self.nonzero_feat,
            edge_types=self.edge_types,
            decoders=self.edge_type2decoder,
        )

    def _optimizer_init(self, batch_size: int, max_margin: float) -> NoReturn:
        """
        Create self.opt.

        Parameters
        ----------
        batch_size : int
            Minibatch size.
        max_margin : float
            Max margin parameter in hinge loss.

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

    def _get_accuracy_scores(self, sess: tf.compat.v1.Session,
                             edges_pos: Dict[Tuple[int, int], List[np.array]],
                             edges_neg: Dict[Tuple[int, int], List[np.array]],
                             edge_type: Tuple[int, int, int]):
        """
        Calculate metrics (AUROC, AUPRC, AP@50)

        Parameters
        ----------
        sess : tf.compat.v1.Session
            Initialized tf session.
        edges_pos : Dict[Tuple[int, int], List[np.array]]
            From edge type to np.arrays of real edges for every edge class in this type.
        edges_neg : Dict[Tuple[int, int], List[np.array]]
            From edge type to np.arrays of fake edges for every edge class in this type.
        edge_type : Tuple[int, int, int]
            Edge type with class.
            Two first elements --- edge type, last element --- class in this type.
        Returns
        -------

        """
        self.feed_dict.update({self.placeholders['dropout']: 0})
        self.feed_dict.update({self.placeholders['batch_edge_type_idx']:
                                   self.minibatch.edge_type2idx[edge_type]})
        self.feed_dict.update({self.placeholders['batch_row_edge_type']: edge_type[0]})
        self.feed_dict.update({self.placeholders['batch_col_edge_type']: edge_type[1]})

        rec = sess.run(self.opt.predictions, feed_dict=self.feed_dict)

        uv = edges_pos[edge_type[:2]][edge_type[2]]
        u = uv[:, 0]
        v = uv[:, 1]
        preds = expit(rec[u, v])
        assert np.all(self.adj_mats[edge_type[:2]][edge_type[2]][u, v] == 1), \
            'Positive examples (real edges) are not exist'

        uv = edges_neg[edge_type[:2]][edge_type[2]]
        u = uv[:, 0]
        v = uv[:, 1]
        preds_neg = expit(rec[u, v])
        assert np.all(self.adj_mats[edge_type[:2]][edge_type[2]][u, v] == 0), \
            'Negative examples (fake edges) are real'

        # Predicted probs
        preds_all = np.hstack([preds, preds_neg])
        # preds_all = np.nan_to_num(preds_all)
        # Real probs: 1 for pos, 0 for neg
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_sc = metrics.roc_auc_score(labels_all, preds_all)
        aupr_sc = metrics.average_precision_score(labels_all, preds_all)

        # Real existing edges (local indexes)
        actual = range(len(preds))
        # All local indexes with probability (sorted)
        predicted = sorted(range(len(preds_all)), reverse=True,
                           key=lambda i: preds_all[i])
        apk_sc = rank_metrics.apk(actual, predicted, k=50)

        return roc_sc, aupr_sc, apk_sc

    def _run_epoch(self, sess: tf.compat.v1.Session, dropout: float,
                   print_progress_every: int, epoch: int, log: bool
                   ) -> NoReturn:
        """
        Run one epoch.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            Initialized tf session.
        dropout : float
            Dropout rate (1 - keep probability).
        print_progress_every : int
            Print statistic every print_progress_every iterations.
        epoch : int
            Number of current epoch (for printing statistic).
        log : bool
            Whether to log or not.
        """
        self.minibatch.shuffle()
        for batch_edges, current_edge_type, current_edge_type_idx in self.minibatch:
            # Construct feed dictionary
            self.feed_dict = self.minibatch.batch_feed_dict(
                batch_edges=batch_edges,
                batch_edge_type=current_edge_type_idx,
                dropout=dropout,
                placeholders=self.placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([self.opt.opt_op, self.opt.cost,
                             self.opt.batch_edge_type_idx],
                            feed_dict=self.feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if self.minibatch.iter % print_progress_every == 0:
                val_auc, val_auprc, val_apk = self._get_accuracy_scores(
                    sess, self.minibatch.val_edges,
                    self.minibatch.val_edges_false,
                    current_edge_type)

                print("Epoch:", "%04d" % (epoch + 1), "Iter:",
                      "%04d" % (self.minibatch.iter + 1), "Edge:", "%04d" % batch_edge_type,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "val_roc=", "{:.5f}".format(val_auc), "val_auprc=",
                      "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk), "time=",
                      "{:.5f}".format(time.time() - t))
                if log:
                    import neptune
                    neptune.log_metric("val_roc", val_auc,
                                       timestamp=time.time())
                    neptune.log_metric("val_apk", val_apk,
                                       timestamp=time.time())
                    neptune.log_metric("val_auprc", val_auprc,
                                       timestamp=time.time())
                    neptune.log_metric("train_loss", train_cost,
                                       timestamp=time.time())

    def run(self, adj_path: str, path_to_split: str, val_test_size: float,
            batch_size: int, num_epochs: int, dropout: float, max_margin: float,
            print_progress_every: int, log: bool, on_cpu: bool, seed: int = 123,
            upload_saved: bool = False) -> NoReturn:
        """
        Run Decagon.

        Parameters
        ----------
        upload_saved : bool
            Default = False
            Whether to log or not.
        adj_path : str
            path for saving/loading adjacency matrices.
        path_to_split : str
            path to save train, test and validate edges.
            If it consist needed edges, they will be loaded.
            Else they will be calculated and saved.
        batch_size : int
            Minibatch size.
        val_test_size : float
            proportion to split edges into train, test and validate.
        num_epochs : int
            number of training epochs.
        dropout : float
            Dropout rate (1 - keep probability).
        print_progress_every : int
            Print statistic every print_progress_every iterations.
        log : bool
            Whether to log or not.
        on_cpu : bool
            Run on cpu instead of gpu.
        max_margin : float
            Max margin parameter in hinge loss.
        seed : int
            Random seed.

        """
        np.random.seed(seed)
        # check if all path exists
        if adj_path and not os.path.exists(adj_path):
            os.makedirs(adj_path)

        if not os.path.exists(path_to_split):
            os.makedirs(path_to_split)

        if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

        if on_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""

        self._adjacency(adj_path)
        self._nodes_features()
        self._edge_types_info()
        self._construct_placeholders()
        self._minibatch_iterator_init(path_to_split, batch_size, val_test_size)
        self._model_init()
        self._optimizer_init(batch_size, max_margin)
        print("Initialize session")
        saver = tf.compat.v1.train.Saver()
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        self.feed_dict = {}

        if upload_saved:
            saver.restore(sess, MODEL_TO_UPLOAD)
            sess.run(tf.compat.v1.global_variables_initializer())
            self.minibatch.shuffle()
            for batch_edges, current_edge_type, current_edge_type_idx in self.minibatch:
                # Construct feed dictionary
                self.feed_dict = self.minibatch.batch_feed_dict(
                    batch_edges=batch_edges,
                    batch_edge_type=current_edge_type_idx,
                    dropout=dropout,
                    placeholders=self.placeholders)
            saver.restore(sess, MODEL_SAVE_PATH)

        dir_to_save_model = f"{MODEL_SAVE_PATH}/model_{datetime.now().isoformat()[:-7]}"
        for epoch in range(num_epochs):
            self._run_epoch(sess, dropout, print_progress_every, epoch, log)
            saver.save(sess, f"{dir_to_save_model}/epoch_{epoch}.ckpt")
        print("Optimization finished!")
        for et in range(self.num_edge_types):
            roc_score, auprc_score, apk_score = self._get_accuracy_scores(
                sess, self.minibatch.test_edges,
                self.minibatch.test_edges_false,
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
            if log:
                import neptune
                neptune.log_metric("ROC-AUC", roc_score)
                neptune.log_metric("AUPRC", auprc_score)
                neptune.log_metric("AP@k score", apk_score)
