import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.heterograph import DGLHeteroGraph
from typing import Dict, Tuple, Callable, Optional


# code from https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero

class RelGraphConvLayer(nn.Module):
    """
    Relational graph convolution layer.

    Parameters
    ----------
    node2in_feat_dim : dict[str, int]
        From node type to corresponding input feature size.
    out_feat_dim : int
        Output feature size.
    rel2nodes : dict[str, tuple[str, str]]
        From edge type (e.g. relation) to begin and end nodes types.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    dropout : float, optional
        Dropout rate. Default: 0.0

    Notes
    -----
    1. About self_loop.
    If you need self_loop (add node embedding from previous layer)
    just make loops in your graph.

    """

    def __init__(self,
                 node2in_feat_dim: Dict[str, int],
                 out_feat_dim: int,
                 rel2nodes: Dict[str, Tuple[str, str]],
                 bias: bool = True,
                 activation: Optional[Callable] = None,
                 dropout: float = 0.0
                 ):
        super(RelGraphConvLayer, self).__init__()
        self.node2in_feat_dim = node2in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.rel2nodes = rel2nodes
        self.rel_names = list(rel2nodes.keys())
        self.bias = bias
        self.activation = activation

        rel2in_feat_dim = {
            rel: self.node2in_feat_dim[rel2nodes[rel][0]]
            for rel in self.rel_names
        }

        # weight = False, because we initialize weights in this class
        # to can adding weights regularization
        # TODO: norm = right? Check it!
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(rel2in_feat_dim[rel], out_feat_dim,
                                  norm='right', weight=False, bias=False)
            for rel in self.rel_names
        })

        # TODO: add weight regularization.
        self.weight = nn.ParameterDict()
        for rel in self.rel_names:
            self.weight[rel] = nn.Parameter(
                torch.Tensor(rel2in_feat_dim[rel], out_feat_dim))
            # TODO: gain = relu? It was good when weights were
            #  initialized together in the 3D tensor.
            nn.init.xavier_uniform_(self.weight[rel],
                                    gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat_dim))
            nn.init.zeros_(self.h_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g: DGLHeteroGraph, node2features: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        Forward computation.

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph. It can be a block.
        node2features : dict[str, torch.Tensor]
            Node features for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        wdict = {rel: {'weight': self.weight[rel]} for rel in self.rel_names}
        hs = self.conv(g, node2features, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
