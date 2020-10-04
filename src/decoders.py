import dgl
import torch
import torch.nn as nn
from typing import List


class HeteroMLPPredictor(nn.Module):
    def __init__(self, embed_dim: int, rel_list: List[str]) -> None:
        """
        Decoder from original decagon article.

        Parameters
        ----------
        embed_dim : int
            Size of embedding
        rel_list : List[str]
            List of all types of nodes. For pairwise side effect edge type
            name should starts with "side".

        """
        super().__init__()
        matrixes = {'R': nn.Linear(embed_dim, embed_dim)}
        params = dict()
        for rel in rel_list:
            if rel[:4] == 'side':
                params[rel] = nn.Parameter(torch.randn(embed_dim))
            else:
                matrixes[rel] = nn.Linear(embed_dim, embed_dim)
        self.matrixes = torch.nn.ModuleDict(matrixes)
        self.params = params

    def apply_edges(self, edges: dgl.udf.EdgeBatch, edge_type: str) -> torch.tensor:
        """
        Apply transformation to all edges, suppose it have type edge_type.

        Parameters
        ----------
        edges : dgl.udf.EdgeBatch
            Edges.
        edge_type : str
            Edge type name.

        Returns
        -------
        torch.tensor
            Score for each edge.
        """
        h_u = edges.src['h']
        h_v = edges.dst['h']
        if (edge_type[:4] == 'side'):
            lft = h_u.mul(self.params[edge_type])
            lft = self.matrixes['R'](lft)
            lft = lft.mul(self.params[edge_type])
        else:
            lft = self.matrixes[edge_type](h_u)
        y = lft.mul(h_v)
        score = torch.sum(y, dim = 1)  
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(lambda edge: self.apply_edges(edge, etype), etype=etype)
            return graph.edges[etype].data['score']

