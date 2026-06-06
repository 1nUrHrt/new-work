import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, scatter


class AttnGIN(MessagePassing):
    def __init__(self, node_feature, edge_feature, h_feature, dp_r, heads):
        super().__init__(aggr="add")
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.h_feature = h_feature
        self.dp_r = dp_r
        self.heads = heads

        assert h_feature % heads == 0
        self.head_dim = h_feature // heads

        self.node_proj = (
            nn.Linear(self.node_feature, self.h_feature)
            if node_feature != h_feature
            else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(self.edge_feature, self.h_feature)
            if edge_feature != h_feature
            else nn.Identity()
        )

        self.eps = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.msg_net = nn.Sequential(
            nn.Linear(self.h_feature * 2, self.h_feature),
            nn.GELU(),
            nn.Linear(self.h_feature, self.h_feature),
        )

        self.attn_net = nn.Linear(self.h_feature * 3, self.heads)

        self.dropout = nn.Dropout(dp_r)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        out = (1 + self.eps) * x + out
        return out

    def message(self, x_i, x_j, edge_attr, edge_index_i):
        tensors = [t for t in (x_j, edge_attr) if t is not None]
        msg_input = torch.cat(tensors, dim=-1)
        msg = self.msg_net(msg_input)
        msg = msg.view(-1, self.heads, self.head_dim)

        tensors = [t for t in (x_i, x_j, edge_attr) if t is not None]
        attn_input = torch.cat(tensors, dim=-1)
        attn_score = self.attn_net(attn_input)
        alpha = softmax(attn_score, edge_index_i, dim=0)
        alpha = self.dropout(alpha)
        alpha = alpha.view(-1, self.heads, 1)
        weighted_msg = msg * alpha
        weighted_msg = weighted_msg.view(-1, self.h_feature)
        return weighted_msg


class AttnGINLayer(nn.Module):
    def __init__(self, hidden_dim, dp_r, heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dp_r = dp_r

        self.attGIN = AttnGIN(hidden_dim, hidden_dim, hidden_dim, dp_r, heads)
        self.LN = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dp_r)

    def forward(self, node, edge_index, edge_attr):
        node = self.LN(node)
        node = self.attGIN(node, edge_index, edge_attr)
        return self.dropout(node)


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


class FFNLayer(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int | None = None, dp_r: float = 0.1):
        super().__init__()
        self.ffn = FFN(d_model, d_ff=d_ff, dp_r=dp_r)
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dp_r)

    def forward(self, x):
        x = self.LN(x)
        x = self.ffn(x)
        return self.dropout(x)


class Readout(nn.Module):
    def __init__(self, in_feature, heads, dp_r):
        super().__init__()
        self.in_features = in_feature
        self.dp_r = dp_r

        assert in_feature % heads == 0
        self.heads = heads
        self.head_dim = in_feature // heads

        self.attn_net = nn.Linear(in_features=in_feature, out_features=self.heads)
        self.dropout = nn.Dropout(self.dp_r)

    def forward(self, nodes, index):
        attn_input = nodes.view(-1, self.heads, self.head_dim)
        attn_score = self.attn_net(nodes)
        alpha = softmax(attn_score, index, dim=0)
        alpha = self.dropout(alpha)
        alpha = alpha.view(-1, self.heads, 1)
        weighted_input = alpha * attn_input
        weighted_input = weighted_input.view(-1, self.in_features)
        graph_emb = scatter(weighted_input, index, dim=0, reduce="sum")
        return graph_emb


class ReadoutBlock(nn.Module):
    def __init__(self, in_feature, heads, dp_r):
        super().__init__()
        self.in_features = in_feature
        self.heads = heads
        self.dp_r = dp_r

        self.LN = nn.LayerNorm(in_feature)
        self.Readout = Readout(in_feature=in_feature, heads=heads, dp_r=dp_r)
        self.BN = nn.BatchNorm1d(in_feature)
        self.ffn = FFN(d_model=in_feature, dp_r=dp_r)
        self.dropout = nn.Dropout(self.dp_r)

    def forward(self, x, index):
        h = self.LN(x)
        x = self.Readout(h, index)

        h = self.BN(x)
        h = self.ffn(h)
        h = self.dropout(h)
        return h + x


class AttnResidual(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))

    def forward(
        self, values: list[torch.Tensor], partial_value: torch.Tensor | None
    ) -> torch.Tensor:
        if partial_value is None:
            arr = []
        else:
            arr = [partial_value]
        V = torch.stack(values + arr, dim=0)  # [L, N, d]
        K = self.norm(V)
        logits = torch.einsum("d,lnd->ln", self.pseudo_query, K)  # [L, N]
        alpha = logits.softmax(dim=0)  # [L, N] — per-node weights over depth
        h = torch.einsum("ln,lnd->nd", alpha, V)  # [N, d]
        return h


class ResTransformerLayer(nn.Module):
    def __init__(
        self,
        h_dim: int,
        dp_r: float,
        heads: int,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.heads = heads
        self.dp_r = dp_r

        self.attn_GIN = AttnGINLayer(h_dim, dp_r=dp_r, heads=heads)

        self.FFN = FFNLayer(h_dim, dp_r=dp_r)

        self.attn_res2GIN = AttnResidual(h_dim)
        self.attn_res2FFN = AttnResidual(h_dim)

    def forward(self, values, partial_value, edge_index, edge_attr):

        h = self.attn_res2GIN(values, partial_value)
        attn_out = self.attn_GIN(h, edge_index, edge_attr)
        if partial_value is None:
            partial_value = attn_out
        else:
            partial_value = partial_value + attn_out

        h = self.attn_res2FFN(values, partial_value)

        partial_value = partial_value + self.FFN(h)

        return partial_value


class AttnResEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        h_dim: int,
        block_num: int,
        dp_r: float,
        heads: int,
        block_size: int = 1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.block_num = block_num
        self.block_size = block_size
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()
        )

        self.res_transformer_layer_list = nn.ModuleList(
            [
                ResTransformerLayer(
                    h_dim=h_dim,
                    dp_r=dp_r,
                    heads=heads,
                )
                for _ in range(block_num)
            ]
        )
        self.final_attn_res = AttnResidual(h_dim)
        self.readout = ReadoutBlock(in_feature=h_dim, dp_r=dp_r, heads=heads)

    def forward(self, batch_data):
        nodes, edge_index, edge_attr, index = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
        )
        nodes = self.node_proj(nodes)
        edge_attr = self.edge_proj(edge_attr)

        values = [nodes]
        partial_value = None
        for i, block in enumerate(self.res_transformer_layer_list):
            partial_value = block(values, partial_value, edge_index, edge_attr)

            if (i + 1) % self.block_size == 0 or i == self.block_num - 1:
                values.append(partial_value)
                partial_value = None
        h = self.final_attn_res(values, partial_value)
        return self.readout(h, index)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, dp_r: float, heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dp_r = dp_r

        self.attn_GIN = AttnGINLayer(hidden_dim, dp_r=dp_r, heads=heads)

        self.ffn = FFNLayer(hidden_dim, dp_r=dp_r)

    def forward(self, node, edge_index, edge_attr):
        h = self.attn_GIN(node, edge_index, edge_attr)
        node = h + node
        h = self.ffn(node)
        return h + node


class AttnEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        h_dim: int,
        block_num: int,
        dp_r: float,
        heads: int,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.block_num = block_num
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()
        )

        self.transformer_layer_list = nn.ModuleList(
            [TransformerLayer(h_dim, dp_r=dp_r, heads=heads) for _ in range(block_num)]
        )

        self.readout = ReadoutBlock(in_feature=h_dim, dp_r=dp_r, heads=heads)

    def forward(self, batch_data):
        nodes, edge_index, edge_attr, index = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
        )
        nodes = self.node_proj(nodes)
        edge_attr = self.edge_proj(edge_attr)

        for encoder in self.transformer_layer_list:
            nodes = encoder(nodes, edge_index, edge_attr)

        return self.readout(nodes, index)


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
