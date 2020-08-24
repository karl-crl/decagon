from __future__ import division
from __future__ import print_function

import gc
import math
from typing import List, Optional, NoReturn, Dict, Tuple

import numpy as np
import scipy.sparse as sp
import os
from copy import deepcopy

from constants import PARAMS
from ..utility import preprocessing

np.random.seed(123)


class EdgeMinibatchIterator(object):
    """
    This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    Attributes:
    ----------
    edges : Dict[Tuple[int, int], int]
        From edge type to number of different classes of these edge type (e.g.
        (0, 0): 1, i.e. protein-protein interaction;
        (1, 1): 3, i.e. 3 classes of drug-drug side effects;
        (0, 1): 1, i.e. protein-drug and (1, 0): 1, i.e. drug-protein).
    num_edge_types : int
        Number of edges types (considering edge classes).
    edge_type2idx : Dict[Tuple[int, int, int], int]
        Enumerate all num_edge_types edges types (considering edge classes).
    idx2edge_type : Dict[int, Tuple[int, int, int]]
        From number to edge type (considering edge classes).
    adj_mats : Dict[Tuple[int, int], List[sp.csr_matrix]]
        From edge type to list of adjacency matrices for each edge class
        (e.g. (1, 1): list of drug-drug adjacency matrices for each se class).
    feat : Dict[int, sp.csr_matrix]
        From edge type (0 = gene, 1 = drug) to feature matrix.
        Row in feature matrix = embedding of one node.
    edge_types : Dict[Tuple[int, int], int]
        From edge type to number of classes of these edge type
        (e. g. (1, 1): number of se).

    train_edges, test_edges, val_edges : Dict[Tuple[int, int], List[np.array]]
        From edge type to list of np.array edges
        (np.array separately for each edge class,
        e.g. (1, 1): [ar1, ar2, ar3], i.e. np.array for each side effect).
    test_edges_false, val_edges_false : Dict[Tuple[int, int], List[np.array]]
        From edge type to list of np.array FAKE edges
        (np.array separately for each edge class,
        e.g. (1, 1): [ar1, ar2, ar3], i.e. np.array for each side effect).

    iter : int
        Number of current iteration.
    freebatch_edge_types : Dict[Tuple[int, int], List[int]]
        For every edge type contains classes,
        what have edges aren't already taken this epoch.
    batch_num : Dict[Tuple[int, int], List[int]]
        From edge type to index of current batch (for every edge class in this type).
    took_all_edges : Dict[Tuple[int, int], int]
        All edges are already taken this epoch?

    """

    def __init__(self,
                 adj_mats: Dict[Tuple[int, int], List[sp.csr_matrix]],
                 feat: Dict[int, sp.csr_matrix],
                 edge_types: Dict[Tuple[int, int], int],
                 path_to_split: str, batch_size: int = PARAMS['batch_size'],
                 val_test_size: float = 0.01, need_sample_edges: bool = False):
        self.adj_mats = adj_mats
        self.feat = feat
        self.edge_types = edge_types
        self.batch_size = batch_size
        self.val_test_size = val_test_size
        self.num_edge_types = sum(self.edge_types.values())

        self.freebatch_edge_types = {edge_type: list(range(edge_class))
                                     for edge_type, edge_class in self.edge_types.items()}
        self.batch_num = {edge_type: [0] * edge_class for edge_type, edge_class in
                          self.edge_types.items()}
        self.took_all_edges = {edge_type: False for edge_type in self.edge_types}
        self.iter = 0

        # Create self.edge_type2idx, self.idx2edge_type
        all_edge_types = [(*edge_type, edge_class)
                          for edge_type, num_classes in edge_types.items()
                          for edge_class in range(num_classes)]
        self.edge_type2idx = dict(zip(all_edge_types,
                                      range(len(all_edge_types))))
        # TODO: May be change dict to list?
        self.idx2edge_type = dict(zip(range(len(all_edge_types)),
                                      all_edge_types))

        if not need_sample_edges:
            self._load_edges(path_to_split)
            return

        edges_init = {edge_type: [None] * n for edge_type, n in
                      self.edge_types.items()}
        self.train_edges = deepcopy(edges_init)
        self.val_edges = deepcopy(edges_init)
        self.test_edges = deepcopy(edges_init)
        self.test_edges_false = deepcopy(edges_init)
        self.val_edges_false = deepcopy(edges_init)
        # Function to build test and val sets with val_test_size positive links
        self.adj_train = deepcopy(edges_init)

        for i, j in self.edge_types:
            for k in range(self.edge_types[i, j]):
                print("Minibatch edge type:", f"({i}, {j}, {k})")
                self.mask_test_edges((i, j), k)
                print("Train edges=", f"{len(self.train_edges[i, j][k]):.4f}")
                print("Val edges=", f"{len(self.val_edges[i, j][k]):.4f}")
                print("Test edges=", f"{len(self.test_edges[i, j][k]):.4f}")
        self._save_edges(path_to_split)

    def _save_edges(self, path_to_split: str) -> NoReturn:
        """
        Save splitted into train/test/validate edges.
        Parameters
        ----------
        path_to_split : str
            Path to saving data.

        Returns
        -------

        """
        print(f'Save edges in {path_to_split}')
        np.save(f'{path_to_split}/train_edges.npy', self.train_edges)
        np.save(f'{path_to_split}/val_edges.npy', self.val_edges)
        np.save(f'{path_to_split}/test_edges.npy', self.test_edges)
        np.save(f'{path_to_split}/test_edges_false.npy', self.test_edges_false)
        np.save(f'{path_to_split}/val_edges_false.npy', self.val_edges_false)
        np.save(f'{path_to_split}/adj_train.npy', self.adj_train)

    def _load_edges(self, path_to_split: str) -> NoReturn:
        """
        Load splitted into train/test/validate edges from files.
        Parameters
        ----------
        path_to_split: str
            Path to load data.

        Returns
        -------

        """
        print(f'Loading edges from {path_to_split}')
        self.train_edges = np.load(f'{path_to_split}/train_edges.npy',
                                   allow_pickle=True).item()
        self.val_edges = np.load(f'{path_to_split}/val_edges.npy',
                                 allow_pickle=True).item()
        self.test_edges = np.load(f'{path_to_split}/test_edges.npy',
                                  allow_pickle=True).item()
        self.test_edges_false = np.load(f'{path_to_split}/' +
                                        f'test_edges_false.npy',
                                        allow_pickle=True).item()
        self.val_edges_false = np.load(f'{path_to_split}/' +
                                       f'val_edges_false.npy',
                                       allow_pickle=True).item()
        # Function to build test and val sets with val_test_size positive links
        self.adj_train = np.load(f'{path_to_split}/' +
                                 f'adj_train.npy',
                                 allow_pickle=True).item()

    @staticmethod
    def preprocess_graph(adj):
        adj = sp.coo_matrix(adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
                degree_mat_inv_sqrt).tocoo()
        else:
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))
            rowdegree_mat_inv = sp.diags(
                np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(
                np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(
                coldegree_mat_inv).tocoo()
        return preprocessing.sparse_to_tuple(adj_normalized)

    @staticmethod
    def _ismember(a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)

    @staticmethod
    def _sample_from_zeros(n: int, sparse: sp.csr_matrix) -> List[List[int]]:
        """
        Sample n zeros from spacse matrix.

        Parameters
        ----------
        n : int
            Number of samples to get from matrix.
        sparse : sp.csr_matrix
            Sparse matrix.

        Returns
        -------
        List[List[int]]
            List of 2-D indices of zeros.

        """
        zeros = np.argwhere(np.logical_not(sparse.todense()))
        ids = np.random.choice(range(len(zeros)), size=(n,))
        return zeros[ids].tolist()

    def _sample_by_row(self, num_of_iters_y: int, sparse: sp.csr_matrix,
                       part_of_zero_i: List[float], batch_size: int,
                       n_of_samples: int, start_idx: int,
                       end_idx: Optional[int] = None) -> list:
        """
        Sample zeros from batch of sparse of kind: sparse[start_idx:end_idx]

        Parameters
        ----------
        num_of_iters_y : int
        sparse : sp.csr_matrix
            Sparse matrix.
        part_of_zero_i : List[float]
            Part on n samples to get from current part of matrix.
        batch_size : int
            Size of batch (height and width).
        n_of_samples : int
            Samples to get from matrix.
        start_idx : int
            Start index of batch by x.
        end_idx : Optional[int]
            End index of batch by x.

        Returns
        -------
        list
            List of samples.
        """
        to_return = []
        for j in range(num_of_iters_y):
            to_sample = math.ceil(n_of_samples * (part_of_zero_i[j]))
            submat = sparse[start_idx:end_idx,
                     j * batch_size:(j + 1) * batch_size]
            ids_in_submat = self._sample_from_zeros(to_sample, submat)
            ids_in_mat = ids_in_submat + \
                         np.array([start_idx, j * batch_size])
            to_return.extend(ids_in_mat)
        j = num_of_iters_y
        if j * batch_size < sparse.shape[1]:
            to_sample = math.ceil(n_of_samples * (part_of_zero_i[j]))
            submat = sparse[start_idx:end_idx,
                     j * batch_size:]
            ids_in_submat = self._sample_from_zeros(to_sample, submat)
            ids_in_mat = ids_in_submat + \
                         np.array([start_idx, j * batch_size])
            to_return.extend(ids_in_mat)
        return to_return

    @staticmethod
    def _get_number_of_zeros_by_row(sparse: sp.csr_matrix, num_of_iters_y: int,
                                    batch_size: int, elements_in_batch: int,
                                    start_idx: int, end_idx: Optional[int] = None
                                    ) -> List[float]:
        """
        Get number of zeros in batch of sparse of kind: sparse[start_idx:end_idx]

        Parameters
        ----------
        sparse : sp.csr_matrix
            Sparse matrix.
        num_of_iters_y : int
        batch_size : int
            Size of batch (height and width).
        elements_in_batch : int
            Number of elements in batch.
        start_idx : int
            Start index of batch by x.
        end_idx : Optional[int]
            End index of batch by x.

        Returns
        -------
        List[float]preprocess_graph
            List of number of zeros in each batch.
        """
        tmp = []
        for j in range(num_of_iters_y):
            tmp.append(1 - sparse[start_idx:end_idx,
                           j * batch_size:(j + 1) * batch_size].count_nonzero()
                       / elements_in_batch)
        j = num_of_iters_y
        if (j * batch_size < sparse.shape[1]):
            sub_mtr = sparse[start_idx:end_idx, j * batch_size:]
            tmp.append(
                1 - sub_mtr.count_nonzero() / (sub_mtr.shape[0] * sub_mtr.shape[1]))
        return tmp

    def _negative_sampling(self, sparse: sp.csr_matrix, n_of_samples: int,
                           batch_size: int = 1000) -> List[List[int]]:
        """
        Perform negative sampling.

        Parameters
        ----------
        sparse : sp.csr_matrix
            Sparse matrix.
        n_of_samples : int
            Number os samples to get.
        batch_size : int
            Size of batch (height and width).

        Returns
        -------
        List[List[int]]
            List of negative samples.
        """
        num_of_iters_x = sparse.shape[0] // batch_size
        num_of_iters_y = sparse.shape[1] // batch_size

        # count nonzero elements on each submatrix
        elements_in_batch = batch_size ** 2
        part_of_zero = []
        for i in range(num_of_iters_x):
            part_of_zero.append(
                self._get_number_of_zeros_by_row(sparse, num_of_iters_y, batch_size,
                                                 elements_in_batch, i * batch_size,
                                                 (i + 1) * batch_size))
        i = num_of_iters_x
        if num_of_iters_x * batch_size < sparse.shape[0]:
            part_of_zero.append(
                self._get_number_of_zeros_by_row(sparse, num_of_iters_y, batch_size,
                                                 elements_in_batch, i * batch_size))

        norm = sum([sum(i) for i in part_of_zero])
        part_of_zero = [[i / norm for i in lst] for lst in part_of_zero]
        result = []
        for i in range(num_of_iters_x):
            print(f"Progress: {i}/{num_of_iters_x}")
            result.extend(self._sample_by_row(num_of_iters_y, sparse, part_of_zero[i],
                                              batch_size, n_of_samples,
                                              i * batch_size, (i + 1) * batch_size))
            gc.collect()
        if num_of_iters_x * batch_size < sparse.shape[0]:
            result.extend(self._sample_by_row(num_of_iters_y, sparse, part_of_zero[i],
                                              batch_size, n_of_samples,
                                              num_of_iters_x * batch_size))
        np.random.shuffle(result)
        return result[:n_of_samples]

    def mask_test_edges(self, edge_type, type_idx):
        # TODO: Make faster
        edges_all, _, _ = preprocessing.sparse_to_tuple(
            self.adj_mats[edge_type][type_idx])
        num_test = max(50,
                       int(np.floor(edges_all.shape[0] * self.val_test_size)))
        num_val = max(50,
                      int(np.floor(edges_all.shape[0] * self.val_test_size)))

        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        val_edges = edges_all[val_edge_idx]

        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges_all[test_edge_idx]

        train_edges = np.delete(edges_all,
                                np.hstack([test_edge_idx, val_edge_idx]),
                                axis=0)

        zeros_to_pick = len(test_edges) + len(val_edges)
        res = self._negative_sampling(self.adj_mats[edge_type][type_idx],
                                      zeros_to_pick)
        test_edges_false = res[:len(test_edges)]  # negative samples
        val_edges_false = res[len(test_edges):]

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats[edge_type][type_idx].shape)
        self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)

        self.train_edges[edge_type][type_idx] = train_edges
        self.val_edges[edge_type][type_idx] = val_edges
        self.val_edges_false[edge_type][type_idx] = np.array(val_edges_false)
        self.test_edges[edge_type][type_idx] = test_edges
        self.test_edges_false[edge_type][type_idx] = np.array(test_edges_false)

    def batch_feed_dict(self, batch_edges, batch_edge_type, dropout, placeholders):
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges})
        feed_dict.update({placeholders['batch_edge_type_idx']: batch_edge_type})
        feed_dict.update({placeholders['batch_row_edge_type']:
                              self.idx2edge_type[batch_edge_type][0]})
        feed_dict.update({placeholders['batch_col_edge_type']:
                              self.idx2edge_type[batch_edge_type][1]})

        feed_dict.update({
            placeholders[f'adj_mats_{i},{j},{k}']: self.adj_train[i, j][k]
            for i, j in self.edge_types for k in range(self.edge_types[i, j])})
        feed_dict.update({placeholders[f'feat_{i}']: self.feat[i]
                          for i, _ in self.edge_types})
        feed_dict.update({placeholders['dropout']: dropout})

        return feed_dict

    def __next__(self) -> Tuple[np.array, Tuple[int, int, int], int]:
        """
        Next batch for train.
        Returns
        -------
        np.array
            Minibatch with train edges.
        Tuple[int, int, int]
            Edge type of minibatch (with edge class, e.g. side effect).
        int
            Index of edge type.

        Notes
        _____
        One epoch = all edges for every type are already taken.
        If all edges are already taken this epoch -> Stop.
        If self.iter % 4 == 0 -> take protein-protein batch.
        If self.iter % 4 == 1 -> take protein-drug batch.
        If self.iter % 4 == 2 -> take drug-protein batch.
        If self.iter % 4 == 3 -> take drug-drug batch from random class.

        self.freebatch_edge_types for every edge type contains classes,
        what have edges aren't already taken this epoch.



        """
        ready_to_take_batch = False
        cur_edge_type, cur_edge_class = None, None
        while not ready_to_take_batch:
            # TODO: self.edge_types --- ordered dict??
            cur_edge_type = list(self.edge_types.keys())[self.iter % len(self.edge_types)]
            if len(self.freebatch_edge_types[cur_edge_type]) == 0:
                # If all edges of current edge type are already taken this epoch
                self.took_all_edges[cur_edge_type] = True
                self.freebatch_edge_types[cur_edge_type] = list(
                    range(self.edge_types[cur_edge_type]))
                self.batch_num[cur_edge_type] = [0] * self.edge_types[cur_edge_type]

            if np.all(list(self.took_all_edges.values())):
                # If all edges are already taken this epoch.
                raise StopIteration

            cur_edge_class = np.random.choice(self.freebatch_edge_types[cur_edge_type])

            # TODO: перепроверить
            cur_batch_num = self.batch_num[cur_edge_type][cur_edge_class]
            out_of_edges = ((cur_batch_num + 1) * self.batch_size - 1 >
                            len(self.train_edges[cur_edge_type][cur_edge_class]))

            if out_of_edges:
                # Go to another class in this edge type (e.g. another side effect)
                self.freebatch_edge_types[cur_edge_type].remove(cur_edge_class)
            else:
                ready_to_take_batch = True

        self.iter += 1
        start = self.batch_num[cur_edge_type][cur_edge_class] * self.batch_size
        self.batch_num[cur_edge_type][cur_edge_class] += 1
        batch_edges = self.train_edges[cur_edge_type][cur_edge_class][
                      start: start + self.batch_size]
        current_edge_type = (*cur_edge_type, cur_edge_class)
        current_edge_type_idx = self.edge_type2idx[current_edge_type]
        return batch_edges, current_edge_type, current_edge_type_idx

    def __iter__(self):
        return self

    def num_training_batches(self, edge_type, type_idx):
        return len(self.train_edges[edge_type][type_idx]) // self.batch_size + 1

    def val_feed_dict(self, edge_type, type_idx, placeholders, size=None):
        edge_list = self.val_edges[edge_type][type_idx]
        if size is None:
            return self.batch_feed_dict(edge_list, edge_type, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, edge_type, placeholders)

    def shuffle(self) -> NoReturn:
        """
        Shuffle train edges and reinitialize self.freebatch_edge_types, self.batch_num,
        self.took_all_edges, self.iter

        Returns
        -------

        """
        for edge_type in self.edge_types:
            for edge_class in range(self.edge_types[edge_type]):
                self.train_edges[edge_type][edge_class] = np.random.permutation(
                    self.train_edges[edge_type][edge_class])

        self.freebatch_edge_types = {edge_type: list(range(edge_class))
                                     for edge_type, edge_class in self.edge_types.items()}
        self.batch_num = {edge_type: [0] * edge_class for edge_type, edge_class in
                          self.edge_types.items()}
        self.took_all_edges = {edge_type: False for edge_type in self.edge_types}
        self.iter = 0
