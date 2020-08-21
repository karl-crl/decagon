import os
from utils import *
from adj_matrix import create_combo_adj, create_adj_matrix
import numpy as np
import scipy.sparse as sp
import pandas as pd
from decagon.utility import preprocessing
from typing import Dict, NoReturn
from run_decagon import RunDecagon


class RunDecagonReal(RunDecagon):
    """
    Decagon runner on real data.


    Attributes
    ----------
    drug_drug_net : nx.Graph
        Drugs as nodes, se as edges.
    combo2stich : Dict[str, np.array]
        From drugs combo name to np.array of two their names.
    combo2se : Dict[str, set]
        From drugs combo name to set of common se of two drugs.
    se2name : Dict[str, str]
        From common se (key) to its real name.
    gene_net : nx.Graph
        Genes (proteins) as nodes, protein-protein-interactions as edges.
    node2idx : Dict[int, int]
        From gene ID (entrez) to its number.
    stitch2se : Dict[str, set]
        From drug (individual) stitch id to a set of its se IDs.
    se2name_mono : Dict[str, str]
        From individual se ID to its real name.
    stitch2proteins : Dict[[str, set]
        From stitch ids (drug) to protein (gene) ids.
    ordered_list_of_drugs : List[str]
        ID of all drugs in mega graph.
    ordered_list_of_se : List[str]
        ID of all se in mega graph.
    ordered_list_of_proteins : List[int]
        Entrez ID of all genes in mega graph.
    self.ordered_list_of_se_mono : List[str]
        ID of all individual  se in drugs embeddings.

    adj_mats : Dict[Tuple[int, int], List[sp.csr_matrix]]
        From edge type to list of adjacency matrices for each edge class
        (e.g. (1, 1): list of drug-drug adjacency matrices for each se class).
        In our case all matrix in adj_mats are symmetric.
    degrees : Dict[int, List[int]]
        Number of connections for each node (0: genes, 1: drugs).
    num_feat : Dict[int, int]
        Number of elements in feature vector for 0: -genes and for 1: -drugs.
    nonzero_feat : Dict[int, int]
        Number of all features for 0: -gene and 1: -drug nodes.
        All features should be nonzero! ????????????
        TODO: What to do with zero features??
        e.g., it is in format 0: num of genes in graph, 1: num of drugs.
    feat : Dict[int, sp.csr_matrix]
        From edge type (0 = gene, 1 = drug) to feature matrix.
        Row in feature matrix = embedding of one node.

    (Other attributes see in parent class)

    """

    def __init__(self, combo_path: str, ppi_path: str, mono_path: str,
                 targets_path: str, min_se_freq: int, min_se_freq_mono: int):
        super().__init__()
        frequent_combo_path = self._leave_frequent_se(combo_path, min_se_freq)
        self.drug_drug_net, self.combo2stitch, self.combo2se, self.se2name = \
            load_combo_se(combo_path=frequent_combo_path)
        self.gene_net, self.node2idx = load_ppi(ppi_path=ppi_path)
        self.stitch2se, self.se2name_mono, se2stitch = load_mono_se(
            mono_path=mono_path)
        self.stitch2proteins = load_targets(targets_path=targets_path)

        self.ordered_list_of_drugs = list(self.drug_drug_net.nodes.keys())
        self.ordered_list_of_se = list(self.se2name.keys())
        self.ordered_list_of_proteins = list(self.gene_net.nodes.keys())

        drugs_set = set(self.ordered_list_of_drugs)
        # Only individual se with frequency > min_se_freq_mono will be saved.
        self.ordered_list_of_se_mono = [
            se_mono for se_mono, stitch_set in se2stitch.items() if
            len(stitch_set.intersection(drugs_set)) > min_se_freq_mono]

    @staticmethod
    def _leave_frequent_se(combo_path: str, min_se_freq: int) -> str:
        """
        Create pre-processed file that only has frequent side effects.

        Parameters
        ----------
        min_se_freq : int
            Only se with frequency >= min_se_freq will be saved.

        Returns
        -------
        str
            Path to combo data considering only frequent se.
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

    def _adjacency(self, adj_path: str) -> NoReturn:
        """
        Create self.adj_mats and self.degrees.

        Parameters
        ----------
        adj_path : str
            Try to use drug-drug adjacency matrices saved in adj_path.
            If this is not possible, calculate it and save in adj_path.

        Notes
        -----
        self.adj_mats : Dict[Tuple[int, int], List[sp.csr_matrix]]
            From edge type to list of adjacency matrices for each edge class
            (e.g. (1, 1): list of drug-drug adjacency matrices for each se class).
            In our case all matrix in adj_mats are symmetric.
        self.degrees : Dict[int, List[int]]
            Number of connections for each node (0: genes, 1: drugs).

        """
        gene_gene_adj = nx.adjacency_matrix(self.gene_net)
        # Number of connections for each gene
        gene_degrees = np.array(gene_gene_adj.sum(axis=0)).squeeze()

        drug_gene_adj = create_adj_matrix(
            a_item2b_item=self.stitch2proteins,
            ordered_list_a_item=self.ordered_list_of_drugs,
            ordered_list_b_item=self.ordered_list_of_proteins)

        gene_drug_adj = drug_gene_adj.transpose(copy=True)

        num_se = len(self.ordered_list_of_se)
        drug_drug_adj_list = []
        try:
            print("Try to load drug-drug adjacency matrices from file.")
            if len(os.listdir(adj_path)) < num_se:
                raise IOError('Not all drug-drug adjacency matrices are saved')
            for i in range(num_se):
                drug_drug_adj_list.append(sp.load_npz(
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
                sp.save_npz(f'{adj_path}/sparse_matrix%04d.npz' % (i,),
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
        # One-hot for genes
        n_genes = self.gene_net.number_of_nodes()
        gene_feat = sp.identity(n_genes)
        gene_nonzero_feat, gene_num_feat = gene_feat.shape
        gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

        # Create sparse matrix with rows -- genes features.
        # Gene feature -- binary vector with length = num of mono se.
        # feature[i] = 1 <=> gene has ith mono se
        drug_feat = create_adj_matrix(
            a_item2b_item=self.stitch2se,
            ordered_list_a_item=self.ordered_list_of_drugs,
            ordered_list_b_item=self.ordered_list_of_se_mono)
        # Check if some gene has zero embedding (i.e. it has no frequent se)
        drugs_zero_features = np.array(
            self.ordered_list_of_drugs)[drug_feat.getnnz(axis=1) == 0]
        # assert 0 not in drug_feat.getnnz(axis=1), \
        # 'All genes should have nonzero embeddings! '
        print(f'Length of drugs features vectors: {drug_feat.shape[1]}')
        print(f'Number of unique vectors: '
              f'{np.unique(drug_feat.toarray(), axis=0).shape[0]}')
        if len(drugs_zero_features) > 0:
            print('Warning! All genes should have nonzero embeddings! ')
            print(f'Where are {len(drugs_zero_features)} zero embeddings')
            print(f'Bad drugs: {drugs_zero_features}')
        drug_nonzero_feat, drug_num_feat = drug_feat.shape
        drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())
        """
        n_drugs = len(self.ordered_list_of_drugs)
        drug_feat = sp.identity(n_drugs)
        drug_nonzero_feat, drug_num_feat = drug_feat.shape
        drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())
    """
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
