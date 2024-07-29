import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    Size,
)
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)

def calc_skew(x):
    m1 = torch.mean(x, axis=0)
    m2 = torch.std(x, axis=0) + 1e-5
    return (m1*m1*m1)/(m2*m2*m2)

def calc_kurtosis(x):
    m1 = torch.mean(x, axis=0)
    m2 = torch.std(x, axis=0) + 1e-5
    return (m1*m1*m1*m1)/(m2*m2*m2*m2)

class SPAConv(MessagePassing):
    def __init__(
            self,
            channels, hidden_size = 16, k = 4,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = channels
        self.out_channels = channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)

        self.src_linear = torch.nn.Linear(k, hidden_size, bias=True)
        self.dst_linear = torch.nn.Linear(k, hidden_size, bias=True)
        self.bias = Parameter(torch.Tensor(channels))
        self.task_query = Parameter(torch.randn(hidden_size))
        self.linear = nn.Linear(channels, channels)
        self.transformer = nn.TransformerEncoderLayer(d_model=k, nhead=1)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.xavier_uniform_(self.src_linear.weight, gain=1.414)
        nn.init.zeros_(self.src_linear.bias)
        nn.init.xavier_uniform_(self.dst_linear.weight, gain=1.414)
        nn.init.zeros_(self.dst_linear.bias)
        zeros(self.bias)

    def forward(self, x, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels
        data = torch.square(x[edge_index][0] - x[edge_index][1]).detach()
        k1 = torch.mean(data, axis=0)
        k2 = torch.std(data, axis=0)
        k3 = calc_skew(data)
        k4 = calc_kurtosis(data)
        S = torch.stack([k1, k2, k3, k4])
        S[torch.isnan(S)] = 0
        S = torch.tanh(S)
        S = self.dropout_layer(S)
        S = F.normalize(S, dim=1).T
        S_src = self.src_linear(S)
        S_dst = self.dst_linear(S)
        att_l = torch.matmul(S_src, self.task_query.unsqueeze(-1)).flatten()
        att_r = torch.matmul(S_dst, self.task_query.unsqueeze(-1)).flatten()

        x_src = x_dst = x.view(-1, H, C)
        x = (x_src, x_dst)

        alpha_src = (x_src * att_l).sum(dim=-1)
        alpha_dst = (x_dst * att_r).sum(-1)
        alpha = (alpha_src, alpha_dst)

        num_nodes = x_src.size(0)
        num_nodes = min(num_nodes, x_src.size(0))
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        out = out.mean(dim=1)
        out = out + self.bias
        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, size_i) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        alpha = F.leaky_relu(alpha, self.negative_slope) * 0.1
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class SPA(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node, heads=1):
        super(SPA, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.fcs = nn.ModuleList([])
        self.gats = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        self.gats.append(SPAConv(hidden_channels, heads=heads))
        for _ in range(self.num_layers - 1):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
            self.gats.append(SPAConv(hidden_channels, heads=heads))
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.gats.append(SPAConv(hidden_channels, heads=heads))
        self.dropout = torch.nn.Dropout(self.dropout)
        self.reset_parameters()
    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)
        for gat in self.gats:
            gat.reset_parameters()


    def forward(self, x, edge_index):
        x = self.dropout(x)
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x)
            x = self.gats[i](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.gats[-1](x, edge_index)
        x = self.fcs[-1](x)
        return x