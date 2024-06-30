import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, add_self_loops
import math

class TransformerEncoder(nn.Module):
    def __init__(self, input_shape, num_heads, dropout, epsilon, ff_dim):
        super(TransformerEncoder, self).__init__()

        # print(input_shape)
        self.MultAtten = nn.MultiheadAttention(
            embed_dim=input_shape[0],
            num_heads=num_heads,
            dropout=dropout,
        )
        self.Dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(input_shape[::-1], epsilon)

        self.conv1d_1 = nn.Conv1d(input_shape[0], ff_dim, 3, 1, 1)
        self.Dropout2 = nn.Dropout(dropout)
        self.act1 = nn.ReLU()

        self.conv1d_2 = nn.Conv1d(ff_dim, input_shape[-1], 3, 1, 1)
        self.act2 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(input_shape[::-1], epsilon)
    def forward(self, inputs):
        # print(inputs.shape)
        inputs = inputs.permute(0, 2, 1)
        x, _ = self.MultAtten(inputs, inputs, inputs)
        x = self.Dropout1(x)
        res = x + inputs
        x = self.layer_norm1(res)

        # print(x.shape)
        # Feed Forward Part
        x = x.permute(0, 2, 1)
        x  = self.conv1d_1(x)
        x = self.Dropout2(x)
        x = self.act1(x)
        # print(x.shape)

        x = self.conv1d_2(x)
        x = self.act2(x)
        x = self.layer_norm2(res)
        res = x + res
        res = res.permute(0, 2, 1)
        return res


class LuongAttention(nn.Module):
    def __init__(self):
        super(LuongAttention, self).__init__()

    def forward(self, query, value, key):
        dim = query.size(-1)
        scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(dim)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attention_weights, value)
        return output, attention_weights
    
class Graph_Water_Transformer_Cov(nn.Module):
    def __init__(self, input_shape, num_heads, dropout, epsilon, lstm_units, num_transformer_blocks, gcn_unit1, gcn_unit2, ff_dim):
        super(Graph_Water_Transformer_Cov, self).__init__()

        self.num_transformer_blocks = num_transformer_blocks
        self.gcn_unit1 = gcn_unit1
        self.gcn_unit2 = gcn_unit2
        self.input_shape = input_shape
        

        self.transformer_encoder = TransformerEncoder(input_shape, num_heads, dropout, epsilon, ff_dim)
        self.fc1 = nn.Linear(input_shape[1], 5)

        self.gcn1 = GCNConv(72, self.gcn_unit1)
        self.gcn2 = GCNConv(self.gcn_unit1, self.gcn_unit2)
        self.gcn_act1 = nn.ReLU()
        self.gcn_act2 = nn.ReLU()



        self.lstm = nn.LSTM(72, lstm_units)
        self.attention = LuongAttention()
        self.flatten = nn.Flatten()
        self.final_fc = nn.Linear(720, 96)
        # self.final_fc = nn.LazyLinear(96)



    def forward(self, cov_inputs, inp_seq, inp_lap):  # inp_seq = tws   inp_lap = edge
        # ======================== covariates with transformer ========================
        cov = cov_inputs
        for _ in range(self.num_transformer_blocks):
            cov = self.transformer_encoder(cov) # (96, 12)

        # print("atten", cov.shape)
        # cov = cov.permute(0, 2, 1) # (12, 96)
        cov = self.fc1(cov) # (96, 5)
        # print("fc1", cov.shape)
        cov_reshape = cov.view(-1, 5, self.input_shape[0])  #(5, 96)
        # print("cov_reshape", cov_reshape.shape)


        # ======================== water levels with GNN ========================
        # GCN
        edge_index = dense_to_sparse(inp_lap)[0]
        edge_index, _ = add_self_loops(edge_index, num_nodes=inp_lap.shape[0])
        x = self.gcn_act1(self.gcn1(inp_seq, edge_index)) #(5, 32)
        x = self.gcn_act2(self.gcn2(x, edge_index))  #(5, 16)

        # LSTM
        xx = self.lstm(inp_seq)[0] #(5, 32)
        # print(cov_reshape.shape, x.shape, xx.shape)
        # ======================== CONCAT and Attention ========================
        x = torch.concat([cov_reshape, x, xx], dim=2)
        x = self.attention(x, x, x)[0]

        x = self.final_fc(self.flatten(x))
        return x