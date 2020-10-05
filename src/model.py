import torch
import torch.nn as nn
from .rgcn import RelGraphConvLayer
from .decoders import HeteroMLPPredictor
from typing import Dict, Tuple
from dgl.heterograph import DGLBlock, DGLHeteroGraph
from typing import List


# TODO: problem with self_loop in RGCN: minibatch consists not all self loops.
#  Solutions:
#  1. Can we add self loops in blocks in model forward?
#  2. Add canonical self_lopp in RGCN -> dimensions problem.
#  Can we pad features vectors with zeros?
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

    def forward(self,
                subgraph: DGLHeteroGraph,
                neg_subgraph: DGLHeteroGraph,
                blocks: List[DGLBlock],
                node2features: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden1(blocks[0], node2features)
        h = self.hidden2(blocks[1], h)
        return self.pred(subgraph, h), self.pred(neg_subgraph, h)
