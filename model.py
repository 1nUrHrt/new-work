import torch
from torch import nn
from torch_geometric.nn import (
    GINConv,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.utils import softmax, scatter
from typing import Literal


class AttnGIN(nn.Module):
    def __init__(self, d_model, dp_r, heads):
        super().__init__()
        self.d_model = d_model
        self.dp_r = dp_r
        self.heads = heads

        assert d_model % heads == 0, (
            f"d_model {d_model} must be divisible by heads {heads}"
        )
        self.head_dim = d_model // heads

        self.msg_proj = nn.Linear(d_model * 2, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dp_r)

    def forward(self, x, edge_index, edge_attr):
        # E:edge num, M:model dim, H:head num, D:head dim, N:node num
        x_j = x[edge_index[0]]  # E,M
        x_i = x[edge_index[1]]  # E,M
        msg_input = self.msg_proj(torch.cat([x_j, edge_attr], dim=-1))  # E,2M -> E,M

        q: torch.Tensor = self.q_proj(x_i).view(
            -1, self.heads, self.head_dim
        )  # E,M -> E,H,D
        k: torch.Tensor = self.k_proj(msg_input).view(
            -1, self.heads, self.head_dim
        )  # E,M -> E,H,D
        v: torch.Tensor = self.v_proj(msg_input).view(
            -1, self.heads, self.head_dim
        )  # E,M -> E,H,D
        attn_score = torch.einsum("ehd,ehd->eh", q, k) / (self.head_dim**0.5)  # E,H,D
        alpha = softmax(attn_score, edge_index[1], dim=0)
        alpha = self.dropout(alpha)
        weighted_v = torch.einsum("eh,ehd->ehd", alpha, v).view(
            -1, self.d_model
        )  # E,H,D -> E,M
        out = scatter(weighted_v, edge_index[1], dim=0, reduce="sum")  # N,M
        return out


class FFN(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int | None = None, dp_r: float = 0.1):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = d_model * 4
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dp_r)

    def forward(self, x):
        x = self.act(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        return self.w3(x)


class AttnGINTransformerLayer(nn.Module):
    def __init__(self, d_model, dp_r, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dp_r = dp_r

        self.LN_e = nn.LayerNorm(d_model)
        self.LN_n = nn.LayerNorm(d_model)
        self.attGIN = AttnGIN(d_model, dp_r, heads)
        self.dropout_a = nn.Dropout(dp_r)

        self.LN_f = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dp_r=dp_r)
        self.dropout_f = nn.Dropout(dp_r)

    def forward(self, node, edge_index, edge_attr):
        h_n = self.LN_n(node)
        h_e = self.LN_e(edge_attr)
        h = self.attGIN(h_n, edge_index, h_e)
        h = self.dropout_a(h)
        node = node + h
        h = self.LN_f(node)
        h = self.ffn(h)
        h = self.dropout_f(h)
        return node + h


class AttnReadout(nn.Module):
    def __init__(self, d_model, heads, dp_r):
        super().__init__()
        self.d_model = d_model
        self.dp_r = dp_r

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.heads = heads
        assert d_model % heads == 0, (
            f"d_model {d_model} must be divisible by heads {heads}"
        )
        self.head_dim = d_model // heads
        self.dropout = nn.Dropout(self.dp_r)

    def forward(self, node, batch_index, graph_attr):
        # N:node num, B:batch size, M:model dim, H:head num, D:head dim
        q: torch.Tensor = self.q_proj(graph_attr[batch_index]).view(
            -1, self.heads, self.head_dim
        )  # N,H,D
        k: torch.Tensor = self.k_proj(node).view(-1, self.heads, self.head_dim)  # N,H,D
        v: torch.Tensor = self.v_proj(node).view(-1, self.heads, self.head_dim)  # N,H,D
        attn_score = torch.einsum("nhd,nhd->nh", q, k) / (self.head_dim**0.5)  # N,H
        alpha = softmax(attn_score, batch_index, dim=0)  # N,H
        alpha = self.dropout(alpha)  # N,H
        weighted_v = torch.einsum("nh,nhd->nhd", alpha, v).view(-1, self.d_model)  # N,M
        return scatter(weighted_v, batch_index, dim=0, reduce="sum")  # B,M


class AttnReadoutTransformerLayer(nn.Module):
    def __init__(self, d_model, heads, dp_r):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dp_r = dp_r

        self.LN_n = nn.LayerNorm(d_model)
        self.LN_g = nn.LayerNorm(d_model)
        self.Readout = AttnReadout(d_model, heads, dp_r)
        self.dropout_r = nn.Dropout(self.dp_r)
        self.LN_f = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, dp_r=dp_r)
        self.dropout_f = nn.Dropout(self.dp_r)

    def forward(self, node, batch_index, graph_attr):
        h_n = self.LN_n(node)
        h_g = self.LN_g(graph_attr)
        h = self.Readout(h_n, batch_index, h_g)
        h = self.dropout_r(h)
        graph_attr = graph_attr + h
        h = self.LN_f(graph_attr)
        h = self.ffn(h)
        h = self.dropout_f(h)
        return h + graph_attr


class AttnGINTFEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        d_model: int,
        block_num: int,
        dp_r: float,
        heads: int,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_dim = graph_dim
        self.d_model = d_model
        self.block_num = block_num
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, d_model) if node_dim != d_model else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, d_model) if edge_dim != d_model else nn.Identity()
        )
        self.graph_proj = (
            nn.Linear(graph_dim, d_model) if graph_dim != d_model else nn.Identity()
        )

        self.attn_gin_tfl_list = nn.ModuleList(
            [
                AttnGINTransformerLayer(d_model, dp_r=dp_r, heads=heads)
                for _ in range(block_num)
            ]
        )

        self.attn_readout_tfl = AttnReadoutTransformerLayer(
            d_model, dp_r=dp_r, heads=heads
        )

    def forward(self, batch_data):
        node, edge_index, edge_attr, index, graph_attr = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
            batch_data.graph_attr,
        )
        node = self.node_proj(node)
        edge_attr = self.edge_proj(edge_attr)
        graph_attr = self.graph_proj(graph_attr)

        for layer in self.attn_gin_tfl_list:
            node = layer(node, edge_index, edge_attr)

        return self.attn_readout_tfl(node, index, graph_attr)


class Classifier(nn.Module):
    def __init__(self, in_feature: int = 256, out_feature: int = 2, dp_r: float = 0.1):
        super().__init__()

        self.in_features = in_feature

        concat_dim = in_feature * 4

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_r),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_r),
            nn.Linear(256, out_feature),
        )

        self._init_weights()

    def forward(self, d1, d2):

        x = torch.cat([d1, d2, torch.abs(d1 - d2), d1 * d2], dim=-1)

        return self.mlp(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EarlyStop:
    def __init__(
        self,
        patience: int = 5,
        mode: Literal["max", "min"] = "max",
        min_delta: float = 1e-4,
    ):
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be either 'max' or 'min', got {mode}")
        self.patience = patience
        self.mode = mode

        self.min_delta = min_delta
        self.counter = 0
        self.best_metric_val = None
        self.early_stop = False

    def __call__(self, metric_value):
        if self.best_metric_val is None:
            is_improved = True
        else:
            if self.mode == "min":
                is_improved = metric_value < self.best_metric_val - self.min_delta
            else:
                is_improved = metric_value > self.best_metric_val + self.min_delta

        if is_improved:
            self.counter = 0
            self.best_metric_val = metric_value
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return is_improved

    def state_dict(self):
        state = {
            "counter": self.counter,
            "best_metric_val": self.best_metric_val,
            "early_stop": self.early_stop,
        }
        return state

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]
        self.best_metric_val = state_dict["best_metric_val"]
        self.early_stop = state_dict["early_stop"]


class GIN(nn.Module):
    def __init__(self, d_model, dp_r):
        super().__init__()
        self.d_model = d_model
        self.dp_r = dp_r
        mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dp_r)
        )
        self.LN = nn.LayerNorm(d_model)
        self.gin = GINConv(mlp, train_eps=False)
        self.dropout = nn.Dropout(dp_r)

    def forward(self, node, edge_index):
        h = self.LN(node)
        h = self.gin(h, edge_index)
        return node + h


class GINEncoder(nn.Module):
    def __init__(self, node_dim, d_model, block_num, dp_r):
        super().__init__()
        self.node_dim = node_dim
        self.d_model = d_model
        self.block_num = block_num
        self.dp_r = dp_r
        self.proj = (
            nn.Linear(node_dim, d_model) if node_dim != d_model else nn.Identity()
        )
        self.gin_list = nn.ModuleList()
        for _ in range(block_num):
            self.gin_list.append(GIN(d_model, dp_r))

        self.readout_proj = nn.Linear(d_model * 2, d_model)
        self.act = nn.ReLU()
        self.readout = nn.Linear(d_model, d_model)

    def forward(self, batch_data):
        node, edge_index, index = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.batch,
        )
        node = self.proj(node)
        for layer in self.gin_list:
            node = layer(node, edge_index)
        out = torch.cat(
            [global_mean_pool(node, index), global_max_pool(node, index)], dim=-1
        )
        out = self.readout_proj(out)
        out = self.act(out)
        return self.readout(out)


__all__ = ["AttnGINTFEncoder", "Classifier", "EarlyStop", "GINEncoder"]
