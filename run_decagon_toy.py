from utils import *
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from decagon.utility import preprocessing
from typing import Dict, NoReturn
from run_decagon import RunDecagon


class RunDecagonToy(RunDecagon):
    """
    Decagon Runner on synthetic data.

    Attributes
    ----------
    n_genes : int
    n_drugs : int
    n_drug_rel_types : int
    gene_net : nx.Graph
    adj_mats : Dict[Tuple[int, int], List[sp.csr_matrix]]
    degrees : Dict[int, List[int]]
    num_feat : Dict[int, int]
    nonzero_feat : Dict[int, int]
    feat : Dict[int, sp.csr_matrix]
    (Other attributes see in parent class)

    """

    def __init__(self, n_genes:int = 500, n_drugs:int = 400,
                 n_drugdrug_rel_types: int = 3):
        super().__init__()
        self.n_genes = n_genes
        self.n_drugs = n_drugs
        self.n_drugdrug_rel_types = n_drugdrug_rel_types
        self.gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)

    def _adjacency(self, adj_path: str = None) -> NoReturn:
        """
        Create self.adj_mats and self.degrees.

        Notes
        -----
        self.adj_mats : Dict[Tuple[int, int], List[sp.csr_matrix]]
            From edge type to list of adjacency matrices for each edge class
            (e.g. (1, 1): list of drug-drug adjacency matrices for each se class).
            In our case all matrix in adj_mats are symmetric.
        self.degrees : Dict[int, List[int]]
            Number of connections for each node (0: genes, 1: drugs).

        """
        gene_adj = nx.adjacency_matrix(self.gene_net)
        gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

        gene_drug_adj = sp.csr_matrix(
            (10 * np.random.randn(self.n_genes, self.n_drugs) > 15).astype(int))
        drug_gene_adj = gene_drug_adj.transpose(copy=True)

        drug_drug_adj_list = []
        tmp = np.dot(drug_gene_adj, gene_drug_adj)
        for i in range(self.n_drugdrug_rel_types):
            mat = np.zeros((self.n_drugs, self.n_drugs))
            for d1, d2 in combinations(list(range(self.n_drugs)), 2):
                if tmp[d1, d2] == i + 4:
                    mat[d1, d2] = mat[d2, d1] = 1.
            drug_drug_adj_list.append(sp.csr_matrix(mat))
        drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for
                             drug_adj in drug_drug_adj_list]
        # data representation
        self.adj_mats = {
            (0, 0): [gene_adj],
            (0, 1): [gene_drug_adj],
            (1, 0): [drug_gene_adj],
            (1, 1): drug_drug_adj_list
        }
        self.degrees = {
            0: [gene_degrees, gene_degrees],
            1: drug_degrees_list + drug_degrees_list,
        }

    def _nodes_features(self) -> NoReturn:
        """
        Create self.num_feat, self.nonzero_feat, self.feat.

        Notes
        -----
        One-hot encoding as genes features.
        Binary vectors with presence of different side effects as drugs features.
        self.num_feat : Dict[int, int]
            Number of elements in feature vector for 0: -genes, for 1: -drugs.
        self.nonzero_feat : Dict[int, int]
            Number of all features for 0: -gene and 1: -drug nodes.
            All features should be nonzero! ????????????
            TODO: What to do with zero features??
            e.g., it is in format 0: num of genes in graph, 1: num of drugs.
        self.feat : Dict[int, sp.csr_matrix]
            From edge type (0 = gene, 1 = drug) to feature matrix.
            Row in feature matrix = embedding of one node.

        """
        # featureless (genes)
        gene_feat = sp.identity(self.n_genes)
        gene_nonzero_feat, gene_num_feat = gene_feat.shape
        gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

        # features (drugs)
        drug_feat = sp.identity(self.n_drugs)
        drug_nonzero_feat, drug_num_feat = drug_feat.shape
        drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

        # data representation
        self.num_feat = {
            0: gene_num_feat,
            1: drug_num_feat,
        }
        self.nonzero_feat = {
            0: gene_nonzero_feat,
            1: drug_nonzero_feat,
        }
        self.feat = {
            0: gene_feat,
            1: drug_feat,
        }
