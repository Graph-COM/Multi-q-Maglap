import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
#from torch.nn import MultiheadAttention
from utils.activation import MultiheadAttention 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

from utils.handle_complex import complex_norm, pairwise_angles

class MultiheadAttentionBias(torch.nn.Module):
    def __init__(self, channels, heads, dropout):
        super(MultiheadAttentionBias, self).__init__()
        self.dropout = dropout
        self.Q = torch.nn.Parameter(torch.randn(channels, channels, heads))
        self.K = torch.nn.Parameter(torch.randn(channels, channels, heads))
        self.V = torch.nn.Parameter(torch.randn(channels, channels, heads))
        torch.nn.init.xavier_uniform(self.Q)
        torch.nn.init.xavier_uniform(self.K)
        torch.nn.init.xavier_uniform(self.V)
        self.bias_Q = torch.nn.Parameter(torch.zeros([channels, heads]))
        self.bias_K = torch.nn.Parameter(torch.zeros([channels, heads]))
        self.bias_V = torch.nn.Parameter(torch.zeros([channels, heads]))
        self.linear_output = torch.nn.Parameter(torch.randn(heads * channels, channels))
        torch.nn.init.xavier_uniform(self.linear_output)

    def forward(self, q, k, v, bias, atten_mask):
        # q,k,v,pe: [batch_size, max_nodes, hidden_size]
        # mask: [batch_size, max_nodes]
        q = torch.einsum('ijh, bnj->bnih', self.Q, q) + self.bias_Q
        k = torch.einsum('ijh, bnj->bnih', self.K, k) + self.bias_K
        key_padding_mask = F._canonical_mask(
            mask=atten_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(atten_mask),
            other_name="attn_mask",
            target_type=q.dtype
        )
        alpha = torch.einsum('bnih, bmih->bnmh', q, k) # q inner products k
        alpha = alpha + key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=-1)
        if bias is not None:
            alpha += bias
        alpha = torch.nn.Softmax(dim=2)(alpha) # softmax normalization
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training) # dropout

        v = torch.einsum('ijh, bnj->bnih', self.V, v) + self.bias_V
        output = torch.einsum('bnmh, bnih->bnih', alpha, v)
        output = output.reshape([output.size(0), output.size(1), -1]) # merge the last two dimension: hidden_size and heads
        output = torch.einsum('ij, bni->bnj', self.linear_output, output) # transform into [batch_size, max_nodes, hidden_size]
        #output = output * (~atten_mask).float().unsqueeze(-1) # mask padding nodes
        return output



class GPSConv(torch.nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.

    .. note::

        For an example of using :class:`GPSConv`, see
        `examples/graph_gps.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graph_gps.py>`_.

    Args:
        channels (int): Size of each input sample.
        conv (MessagePassing, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        attn_dropout (float, optional): Dropout probability of the normalized
            attention coefficients. (default: :obj:`0`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
    """
    def __init__(
            self,
            channels: int,
            pe_input_dim: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            symmetry: str = None,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.symmetry = symmetry



        # TO DO: merge standard attention into our one;
        # TO DO: add dropout
        if self.symmetry == 'invariance':
            # a customized attention with atten bias term
            self.attn = MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True
            )
        elif self.symmetry is None:
            self.attn = MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True
            )
            #self.attn = MultiheadAttentionBias(
            #    channels,
            #    heads,
            #    dropout=attn_dropout
            #)

        if self.symmetry == 'invariance':
            self.pe_bias_project_layer = Sequential(
            Linear(2*pe_input_dim, pe_input_dim),
            activation_resolver(act, **(act_kwargs or {})),
            Linear(pe_input_dim, heads),
        )


        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            pe: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)
        if self.symmetry == 'invariance':
            pe, _ = to_dense_batch(pe, batch)
            pe_norm = complex_norm(pe)
            real, imag = pairwise_angles(pe)
            bias = [pe_norm.unsqueeze(2).expand([-1, -1, pe_norm.size(1), -1]),
                            pe_norm.unsqueeze(1).expand([-1, pe_norm.size(1), -1, -1]), real, imag]
            bias = torch.cat(bias, dim=-1)
            #pe_difference = pe.unsqueeze(2).expand([-1, -1, pe.size(1), -1]) - \
            #                pe.unsqueeze(1).expand([-1, pe.size(1), -1, -1])
            #bias = complex_norm(pe_difference)
            bias = self.pe_bias_project_layer(bias)
            bias = torch.flatten(bias.transpose(1, -1), 0, 1)
            # h, _ = self.attn(h, h, h, bias, ~mask)
        else:
            bias = None
            # h, _ = self.attn(h, h, h, bias, ~mask)
            # h = self.attn(h, h, h, bias, ~mask)

        h, _ = self.attn(h, h, h, bias, ~mask)
        #h, _ = self.attn(h, h, h, ~mask)
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')



