import scipy.sparse as sp
import numpy as np
from typing import List


def create_adj_matrix(a_item2b_item: dict,
                      ordered_list_a_item: list,
                      ordered_list_b_item: list) -> sp.csr_matrix:
    """

    Parameters
    ----------
    a_item2b_item: dict
        dictionary from a_item to set of corresponding b_items
    ordered_list_a_item: list
        list with required a_items (unique)
    ordered_list_b_item: list
        list with required b_items (unique)

    Returns
    -------
    sp.csc_matrix
        adjacency matrix for a_items and b_items.
        adjacency matrix[i, j] = 1 <=> b_item in a_item2b_item[a_item]
        (a_item = ordered_list_a_item[i], b_item = ordered_list_b_item[i])

    """
    num_a_item = len(ordered_list_a_item)
    num_b_item = len(ordered_list_b_item)
    b_item2idx = dict(zip(ordered_list_b_item, range(num_b_item)))
    rows = []  # Non zero row indexes in adj_matrix
    cols = []  # Non zero column indexes in adj_matrix
    for i, a_item in enumerate(ordered_list_a_item):
        correspond_b_items = a_item2b_item[a_item].intersection(
            set(ordered_list_b_item))
        for b_item in correspond_b_items:
            rows.append(i)
            cols.append(b_item2idx[b_item])

    data = np.ones_like(rows)
    adj_matrix = sp.csr_matrix((data, (rows, cols)),
                               shape=(num_a_item, num_b_item))
    return adj_matrix


def create_combo_adj(combo_a_item2b_item: dict,
                     combo_a_item2a_item: dict,
                     ordered_list_a_item: list,
                     ordered_list_b_item: list) -> List[sp.csr_matrix]:
    """

    Parameters
    ----------
    combo_a_item2b_item: dict
         dict from combo a_items name to set of corresponding b_items
         (e.g. drug-drug pair name -> names of corresponding side effects)
    combo_a_item2a_item: dict
        dict from combo a_items name to sequence of two individual a_item names
        (e.g. combo drug-drug pair name -> np.array of two individual names)
    ordered_list_a_item: list
        list with required a_items (unique)
        (e.g. ordered list of drugs)
    ordered_list_b_item: list
        list with required b_items (unique)
        (e.g. ordered list of combo se)

    Returns
    -------
    List[sp.csr_matrix]
        list of adjacency matrices for a_items.
        For each b_item this matrix creates separately.
        (e.g. drug-drug adjacency matrix for each side effect.
        se adjacency matrix element is nonzero for two drugs <=>
        <=> corresponding two drugs have this se)


    """
    num_a_item = len(ordered_list_a_item)
    num_b_item = len(ordered_list_b_item)
    a_item2idx = dict(zip(ordered_list_a_item, range(num_a_item)))
    b_item2idx = dict(zip(ordered_list_b_item, range(num_b_item)))

    # rows[i], cols[i] -- lists of non zero indexes in adj_matrix of ith b_item
    rows = [[] for _ in range(num_b_item)]
    cols = [[] for _ in range(num_b_item)]
    for combo_a_item, set_b_item in combo_a_item2b_item.items():
        a_item1, a_item2 = combo_a_item2a_item[combo_a_item]
        if a_item1 not in ordered_list_a_item:
            continue
        if a_item2 not in ordered_list_a_item:
            continue
        idx_a_item1 = a_item2idx[a_item1]
        idx_a_item2 = a_item2idx[a_item2]
        correspond_b_items = combo_a_item2b_item[combo_a_item].intersection(
            set(ordered_list_b_item))
        for b_item in correspond_b_items:
            idx_b_item = b_item2idx[b_item]

            rows[idx_b_item].append(idx_a_item1)
            cols[idx_b_item].append(idx_a_item2)

            rows[idx_b_item].append(idx_a_item2)
            cols[idx_b_item].append(idx_a_item1)
    drug_drug_adj_list = [sp.csr_matrix((np.ones_like(rs), (rs, cs)),
                                        shape=(num_a_item, num_a_item))
                          for rs, cs in zip(rows, cols)]
    return drug_drug_adj_list
