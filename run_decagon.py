import os
from utils import *
from adj_matrix import create_combo_adj, create_adj_matrix
import numpy as np
import scipy.sparse as sp
import pandas as pd
from decagon.utility import preprocessing
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator



class RunDecagon:
    """
    Attributes
    ----------
    drug_drug_net: nx.Graph
        drugs as nodes, se as edges.
    combo2stich: Dict[str, np.array]
        from drugs combo name to np.array of two their names.
    combo2se: Dict[str, set]
        from drugs combo name to set of common se of two drugs.
    se2name: Dict[str, str]
        from common se (key) to its real name.
    gene_net: nx.Graph
        genes (proteins) as nodes, protein-protein-interactions as edges
    node2idx: Dict[int, int]
        from gene ID (entrez) to its number
    stitch2se: Dict[str, set]
        from drug (individual) stitch id to a set of its se IDs
    se2name_mono: Dict[str, str]
        from individual se ID to its real name
    stitch2proteins: Dict[[str, set]
        from stitch ids (drug) to protein (gene) ids

    ordered_list_of_drugs: List[str]
        ID of all drugs in mega graph
    ordered_list_of_se: List[str]
        ID of all se in mega graph
    ordered_list_of_proteins: List[int]
        Entrez ID of all genes in mega graph
    self.ordered_list_of_se_mono: List[str]
        ID of all individual  se in drugs embeddings

    adj_mats: Dict[Tuple[int, int], List[sp.csr_matrix]]
        from edge type to list of adjacency matrices for each edge class
        (e.g. (1, 1): list of drug-drug adjacency matrices for each se class)
        In our case all matrix in adj_mats are symmetric
    degrees: Dict[int, List[int]]
        number of connections for each node (0: genes, 1: drugs)

    edge_type2dim: Dict[Tuple[int, int], List[int]
        from edge type to list of shapes all its adjacency matrices.
    edge_type2decoder: Dict[Tuple[int, int], str]
        from edge type to decoder type
        (we use different decompositions for different edges types)
    edge_types: Dict[Tuple[int, int], int]
        from edge type to number of classes of these edge type
        (e. g. (1, 1): number of se)
    num_edge_types: int
        number of all edge types (considering all classes)

    num_feat: Dict[int, int]
        number of elements in feature vector for 0: -genes and for 1: -drugs.
    nonzero_feat: Dict[int, int]
        number of all features for 0: -gene and 1: -drug nodes.
        All features should be nonzero!
        e.g., it is in format 0: num of genes in graph, 1: num of drugs.
    feat: Dict[int, sp.csr_matrix]
        from edge type (0 = gene, 1 = drug) to feature matrix.
        row in featire matrix = embedding of one node.



    """

    def __init__(self, combo_path: str, ppi_path: str, mono_path: str,
                 targets_path: str, min_se_freq: int):
        frequent_combo_path = self._leave_frequent_se(combo_path, min_se_freq)
        self.drug_drug_net, self.combo2stitch, self.combo2se, self.se2name = \
            load_combo_se(combo_path=frequent_combo_path)
        self.gene_net, self.node2idx = load_ppi(ppi_path=ppi_path)
        self.stitch2se, self.se2name_mono = load_mono_se(mono_path=mono_path)
        self.stitch2proteins = load_targets(targets_path=targets_path)

        self.ordered_list_of_drugs = list(self.drug_drug_net.nodes.keys())
        self.ordered_list_of_se = list(self.se2name.keys())
        self.ordered_list_of_proteins = list(self.gene_net.nodes.keys())
        self.ordered_list_of_se_mono = list(self.se2name_mono.keys())

    @staticmethod
    def _leave_frequent_se(combo_path: str, min_se_freq: int) -> str:
        """
        Create pre-processed file that only has frequent side effects
        Parameters
        ----------
        min_se_freq: int
            Only se with frequency >= min_se_freq will be saved.

        Returns
        -------
        str
            path to combo data considering only frequent se.
        """
        all_combo_df = pd.read_csv(combo_path)
        se_freqs = all_combo_df["Polypharmacy Side Effect"].value_counts()
        frequent_se = se_freqs[se_freqs >= min_se_freq].index.tolist()
        frequent_combo_df = all_combo_df[
            all_combo_df["Polypharmacy Side Effect"].isin(frequent_se)]

        filename, file_extension = os.path.splitext(combo_path)
        frequent_combo_path = filename + '-freq-only' + file_extension
        frequent_combo_df.to_csv(frequent_combo_path, index=False)
        return frequent_combo_path

    def _adjacency(self, adj_path: str) -> None:
        gene_gene_adj = nx.adjacency_matrix(self.gene_net)
        # Number of connections for each gene
        gene_degrees = np.array(gene_gene_adj.sum(axis=0)).squeeze()

        drug_gene_adj = create_adj_matrix(
            a_item2b_item=self.stitch2proteins,
            ordered_list_a_item=self.ordered_list_of_drugs,
            ordered_list_b_item=self.ordered_list_of_proteins)

        gene_drug_adj = drug_gene_adj.transpose(copy=True)

        num_se = len(self.ordered_list_of_se)
        if not os.path.isdir(adj_path):
            os.mkdir(adj_path)
        drug_drug_adj_list = []
        try:
            print("Try to load drug-drug adjacency matrices from file.")
            if len(os.listdir(adj_path)) < num_se:
                raise IOError('Not all drug-drug adjacency matrices are saved')
            for i in range(num_se):
                drug_drug_adj_list[i].append(sp.load_npz(
                    adj_path + '/sparse_matrix%04d.npz' % i).tocsr())
        except IOError:
            print('Calculate drug-drug adjacency matrices')
            drug_drug_adj_list = create_combo_adj(
                combo_a_item2b_item=self.combo2se,
                combo_a_item2a_item=self.combo2stitch,
                ordered_list_a_item=self.ordered_list_of_drugs,
                ordered_list_b_item=self.ordered_list_of_se)
            print("Saving matrices to file")
            for i in range(len(drug_drug_adj_list)):
                sp.save_npz('adjacency_matrices/sparse_matrix%04d.npz' % (i,),
                            drug_drug_adj_list[i].tocoo())
        # Number of connections for each drug
        drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze()
                             for drug_adj in drug_drug_adj_list]
        self.adj_mats = {
            (0, 0): [gene_gene_adj],
            (0, 1): [gene_drug_adj],
            (1, 0): [drug_gene_adj],
            (1, 1): drug_drug_adj_list,
        }
        self.degrees = {
            0: [gene_degrees],
            1: drug_degrees_list,
        }

        def _nodes_features(self) -> None:
            # One-hot for genes
            n_genes = self.gene_net.number_of_nodes()
            gene_feat = sp.identity(n_genes)
            gene_nonzero_feat, gene_num_feat = gene_feat.shape
            # TODO: check this function
            gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

            # Create sparse matrix with rows -- genes features.
            # Gene feature -- binary vector with length = num of mono se.
            # feature[i] = 1 <=> gene has ith mono se
            drug_feat = create_adj_matrix(
                a_item2b_item=self.stitch2se,
                ordered_list_a_item=self.ordered_list_of_drugs,
                ordered_list_b_item=self.ordered_list_of_se_mono)
            # Check if some gene has zero embedding (i.e. it has no frequent se)
            assert 0 not in drug_feat.getnnz(axis=1), \
                'All genes should have nonzero embeddings! '
            drug_nonzero_feat, drug_num_feat = drug_feat.shape
            drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

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

        def _edge_types_info(self) -> None:
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
                                     val_test_size: float):
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



