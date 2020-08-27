from __future__ import division
from __future__ import print_function

import gc
import math
from typing import List, Optional, NoReturn, Dict, Tuple, Any

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
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
        (0, 1): 1, i.e. protein-drug;
        (1, 0): 1, i.e. drug-protein).
    symmetry_types_groups : List[List]
        Should contains lists with len in {1, 2}.
        All types of edges splits into groups of symmetry.
        E. g. symmetry_types_groups = [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]].
        Two types from one group of symmetry have same edges, differing only in direction
        (e.g (0, 1) has protein -> drug edges and (1, 0) has drug -> protein edges).
        It is needed to good splitting edges into train/validate/test,
        because independent splitting of symmetry classes leads to leakage problem
        (e.g. protein_A -> drug_B in train, drug_B -> protein_A in test).
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

    adj_train: Dict[Tuple[int, int], List[sp.csr_matrix]]
        From edge type to list of train normalized adjacency matrices for each edge class
        in this type.

    iter : int
        Number of current iteration.
    freebatch_edge_types : Dict[Tuple[int, int], List[int]]
        For every edge type contains classes,
        what have edges aren't already taken.
    batch_num : Dict[Tuple[int, int], List[int]]
        From edge type to index of current batch (for every edge class in this type).
    took_all_edges : Dict[Tuple[int, int], int]
        All edges are already taken this epoch? (For every edge type).
    ordered_edge_types: List[Tuple[int, int]]
        List of all edge types.

    val_test_size : float
        Proportion of train and validate data. It should be < 0.5!
    batch_size : int
        Minibatch size.

    """

    def __init__(self,
                 adj_mats: Dict[Tuple[int, int], List[sp.csr_matrix]],
                 feat: Dict[int, sp.csr_matrix],
                 edge_types: Dict[Tuple[int, int], int],
                 symmetry_types_groups: List[List],
                 path_to_split: str, batch_size: int = PARAMS['batch_size'],
                 val_test_size: float = 0.01, need_sample_edges: bool = False):

        self.adj_mats = adj_mats
        self.feat = feat
        self.edge_types = edge_types
        self.symmetry_types_groups = symmetry_types_groups
        self.ordered_edge_types = list(self.edge_types.keys())
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
        self.adj_train = deepcopy(edges_init)

        for types_group in self.symmetry_types_groups:
            first_edge_type = types_group[0]
            for edge_class in range(self.edge_types[first_edge_type]):
                print("Minibatch edge type:", f"({first_edge_type}, {edge_class})")
                self._mask_test_edges(first_edge_type, edge_class)
                self._train_adjacency(first_edge_type, edge_class)
                print("Train edges=",
                      f"{len(self.train_edges[first_edge_type][edge_class]):.4f}")
                print("Val edges=",
                      f"{len(self.val_edges[first_edge_type][edge_class]):.4f}")
                print("Test edges=",
                      f"{len(self.test_edges[first_edge_type][edge_class]):.4f}")

                if len(types_group) > 1:
                    second_edge_type = types_group[1]
                    self._make_symmetry_edges(edge_type=first_edge_type,
                                              symmetry_edge_type=second_edge_type,
                                              edge_class=edge_class)
                    # TODO: may be just take adj for first_edge_type and transpose it?
                    self._train_adjacency(second_edge_type, edge_class)

        self._save_edges(path_to_split)

    @staticmethod
    def _inverse_edges(edges: np.array) -> np.array:
        """
        Inverse all given edges.
        Parameters
        ----------
        edges : np.array
            Edges in format --- one row = one nodes pair.

        Returns
        -------
        np.array
            Inverse edges (e.g. edge [1, 2] -> [2, 1]).

        """
        inversed_edges = edges.copy()
        inversed_edges[:, [0, 1]] = inversed_edges[:, [1, 0]]
        return inversed_edges

    def _make_symmetry_edges(self, edge_type: Tuple[int, int],
                             symmetry_edge_type: Tuple[int, int],
                             edge_class: int) -> NoReturn:
        """
        Inverse train, validate, test, neg. validate and neg. test edges
        of given edge_type and edge_class to make split edges of symmetry edge_type
        (and same edge_class).

        Parameters
        ----------
        edge_type : Tuple[int, int]
            Type of edges.
        symmetry_edge_type : Tuple[int, int]
            Type of edges in one symmetry group with edge_type
            (e.g. if edge_type = (0, 1), symmetry edge_type can be (1, 0)).
        edge_class : int
            Index of edge class.

        Returns
        -------

        """
        all_edges_splits = [self.train_edges, self.val_edges, self.test_edges,
                            self.val_edges_false, self.test_edges_false]
        for edges_split in all_edges_splits:
            edges_split[symmetry_edge_type][edge_class] = self._inverse_edges(
                edges_split[edge_type][edge_class])

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
        self.adj_train = np.load(f'{path_to_split}/' +
                                 f'adj_train.npy',
                                 allow_pickle=True).item()

    @staticmethod
    def preprocess_graph(adj: sp.csr_matrix, same_type_nodes: bool = True
                         ) -> Tuple[np.array, np.array, Tuple[int, int]]:
        """

        Parameters
        ----------
        adj : sp.csr_matrix
            Adjacency matrix.
        same_type_nodes : bool
            Is adjacency matrix for nodes of same type or not?
            E.g. drug-drug, protein-protein adj or drug-protein, protein-drug.

        Returns
        -------
        np.array
            Pairs of edges in normalized adjacency matrix with nonzero values.
        np.array
            Nonzero values of normalized adjacency matrix.
        Tuple[int, int]
            Shape of normalized adjacency matrix.

        Notes
        -----
        Updating embeddings on new layer can be written as
        H(l+1) = σ(SUM_r A_r_normalize @ H(l) @ W_r(l))
        A_r_normalize --- normalized adj matrix for r edge type.

        So we have two variants of normalization for A_r (further just A).
        1. Adj matrix for nodes of same type. It is symmetric.
            A_ = A + I,
            to add information of current node when collecting information from neighbors
            with same type.
            E.g. collecting info from drug nodes when update current drug embedding.

            D: degree matrix (diagonal matrix with number of neighbours on the diagonal).
            A_normalize = D^(-1/2) @ A_ @ D^(-1/2),
            to symmetric normalization (division by sqrt(N_r^i) * sqrt(N_r^j)
            in formula from original paper).

        2. Adj matrix for nodes of different type.
            Here we don't need to add information from the current node.

            D_row: output degree matrix --- diagonal matrix with number of output
            neighbours (i.e. i -> neighbours) on the diagonal.
            D_col: input degree matrix --- diagonal matrix with number of input
            neighbours (i.e. neighbours -> i) on the diagonal.
            A_normalize = D_row^(-1/2) @ A @ D_col^(-1/2),
            to symmetric normalization (division by sqrt(N_r^i) * sqrt(N_r^j)
            in formula from original paper).


        """
        adj = sp.coo_matrix(adj)
        if same_type_nodes:
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
        List[float]
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

    def _mask_test_edges(self, edge_type: Tuple[int, int], edge_class: int,
                         min_val_test_size: int = 50) -> None:
        """
        Split edges into train/validate/test.
        Write into self.adj_train[edge_type][edge_class]
        normalized train adjacency matrix.

        Parameters
        ----------
        edge_type : Tuple[int, int]
            Type if edges.
        edge_class : int
            Index of edge class in given edge type
            (e. g. for (1, 1) i means ith side effect).
        min_val_test_size : int
            Minimum size of test and validation samples.

        Returns
        -------

        """
        if self.val_test_size >= 0.5:
            print('proportions of validation and test data should be < 0.5')
            raise ValueError

        # Split positive examples.

        # np.array with pairs of nodes (edges with current edge_type and edge_class).
        edges_all, _, _ = preprocessing.sparse_to_tuple(
            self.adj_mats[edge_type][edge_class])
        num_test = max(min_val_test_size,
                       int(np.floor(edges_all.shape[0] * self.val_test_size)))
        num_val = max(min_val_test_size,
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

        self.train_edges[edge_type][edge_class] = train_edges
        self.val_edges[edge_type][edge_class] = val_edges
        self.test_edges[edge_type][edge_class] = test_edges

        # Split negative examples.

        zeros_to_pick = len(test_edges) + len(val_edges)
        res = self._negative_sampling(self.adj_mats[edge_type][edge_class],
                                      zeros_to_pick)
        test_edges_false = res[:len(test_edges)]
        val_edges_false = res[len(test_edges):]

        self.val_edges_false[edge_type][edge_class] = np.array(val_edges_false)
        self.test_edges_false[edge_type][edge_class] = np.array(test_edges_false)

    def _train_adjacency(self, edge_type: Tuple[int, int], edge_class: int
                         ) -> NoReturn:
        """
        Create train adjacency matrix for given edge type and class, normalize it.

        Parameters
        ----------
        edge_type : Tuple[int, int]
            Type of edges.
        edge_class : int
            Index of edge class in given type.

        Returns
        -------

        """
        # Adjacency matrix for train edges (1 correspond train edges).
        train_edges = self.train_edges[edge_type][edge_class]
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats[edge_type][edge_class].shape)
        # Normalize train adjacency matrix.
        self.adj_train[edge_type][edge_class] = self.preprocess_graph(
            adj_train, same_type_nodes=(edge_type[0] == edge_type[1]))

    def batch_feed_dict(self, batch_edges: np.array,
                        batch_edge_type: int, dropout: float,
                        placeholders: Dict[str, tf.compat.v1.placeholder]
                        ) -> Dict[str, Any]:
        """
        Create feed dict for minibatch.
        Parameters
        ----------
        batch_edges : np.array
            Minibatch with train edges.
        batch_edge_type : int
            Index of type of batch edges (with class in this type)
            (index in edge_type2idx).
        dropout : float
            Dropout rate (1 - keep probability).
        placeholders : Dict[str, tf.compat.v1.placeholder]
            Variables for input data of decagon model.

        Returns
        -------
        Dict[str, Any]
            Feed dict for minibatch (feed dict --- values of placeholders).

        """
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
        If self.iter % 4 == 3 -> take drug-drug batch from random side effect.
        """
        cur_edge_type = self.ordered_edge_types[self.iter % len(self.edge_types)]
        # Remove from freebatch_edge_types classes, what are already fully taken.
        self.freebatch_edge_types[cur_edge_type] = [
            edge_class for edge_class in self.freebatch_edge_types[cur_edge_type]
            if self.batch_num[cur_edge_type][edge_class] + 1 <= self.num_training_batches(
                cur_edge_type, edge_class)]

        # If we take all edges from current type, but another type is not fully taken,
        # we should to restart getting edges from current type.
        if len(self.freebatch_edge_types[cur_edge_type]) == 0:
            # We take all edges from current type.
            self.took_all_edges[cur_edge_type] = True
            # All edges from current type (from all classes) are ready to taken again.
            self.freebatch_edge_types[cur_edge_type] = list(
                range(self.edge_types[cur_edge_type]))
            # Make zero index of batches.
            self.batch_num[cur_edge_type] = [0] * self.edge_types[cur_edge_type]
        # If all edges are already taken this epoch.
        if np.all(list(self.took_all_edges.values())):
            raise StopIteration
        # Select random class from current edge type to sample batch
        # (e.g. select specific side effect for drug-drug edges).
        cur_edge_class = np.random.choice(self.freebatch_edge_types[cur_edge_type])
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

    def num_training_batches(self, edge_type: Tuple[int, int], edge_class: int) -> int:
        """
        Number of different training batches for given edge type and
        class in this type.
        Parameters
        ----------
        edge_type : Tuple[int, int]
            Edge type (e.g. (0, 0) for protein-protein edges, (1, 1) for drug-drug).
        type_idx : int
            Edge class in given type (e.g. index of side effect for (1, 1)).

        Returns
        -------
        int
            Number og batches.

        Notes
        -----
        Only batches with batch_size can be given into model.
        If last batch is smaller, it skips.

        """
        return len(self.train_edges[edge_type][edge_class]) // self.batch_size

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
        self.took_all_edges, self.iter.

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
