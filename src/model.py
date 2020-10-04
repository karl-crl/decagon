import torch
import torch.nn as nn
from .rgcn import RelGraphConvLayer
from .decoders import HeteroMLPPredictor
from typing import Dict, Tuple
import dgl


# Toy model.
# TODO: Add minibatch (blocks).
class Model(nn.Module):
    def __init__(self, node2in_feat_dim: Dict[str, int],
                 hidden_dim: int, embed_dim: int,
                 rel2nodes: Dict[str, Tuple[str, str]],
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        node_names = node2in_feat_dim.keys()

        self.hidden1 = RelGraphConvLayer(node2in_feat_dim=node2in_feat_dim,
                                         out_feat_dim=hidden_dim,
                                         rel2nodes=rel2nodes,
                                         bias=bias,
                                         dropout=dropout,
                                         activation=torch.nn.ReLU())

        self.hidden2 = RelGraphConvLayer(
            node2in_feat_dim={node: hidden_dim for node in node_names},
            out_feat_dim=embed_dim,
            rel2nodes=rel2nodes,
            bias=bias,
            dropout=dropout,
            activation=None)
        self.pred = HeteroMLPPredictor(embed_dim, list(rel2nodes.keys()))

    def forward(self, g: dgl.heterograph, neg_g: dgl.heterograph,
                node2features: Dict[str, torch.Tensor], etype: str):
        h = self.hidden1(g, node2features)
        h = self.hidden2(g, h)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)
