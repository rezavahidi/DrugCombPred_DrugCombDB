
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv,  GATConv, TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

# a simple base GCN model
# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         torch.manual_seed(1234)
#         self.conv = GCNConv(in_channels, out_channels, add_self_loops=False)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv(x, edge_index, edge_weight).relu()
#         return x
    

# base from this notebook: https://colab.research.google.com/drive/1LJir3T6M6Omc2Vn2GV2cDW_GV2YfI53_?usp=sharing#scrollTo=jNsToorfSgS0
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels): # num_features = dataset.num_features
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_features)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()

        return x


# model = GCN(dataset.num_features, dataset.num_classes)
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# print("Training on CPU.")

# for epoch in range(1, 6):
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, data.edge_attr)
#     loss = F.cross_entropy(out, data.y)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch}, Loss: {loss}")


class TransformerGNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(TransformerGNN, self).__init__()
        embedding_size = model_params["model_embedding_size"] # default: 1024
        n_heads = model_params["model_attention_heads"] # default: 3
        self.n_layers = model_params["model_layers"] # default: 3
        dropout_rate = model_params["model_dropout_rate"] # default: 0.3
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.leakyrelu = nn.LeakyReLU(0.1)

        # Transformation layer
        self.conv1 = GATConv(feature_size, 
                                    embedding_size, 
                                    heads=n_heads, 
                                    dropout=dropout_rate,
                                    edge_dim=edge_dim) 

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(GATConv(embedding_size, 
                                                    embedding_size, 
                                                    heads=n_heads, 
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        # test extra layer ---------------------------
        # self.conv2 = TransformerConv(embedding_size * n_heads, 
        #                             embedding_size, 
        #                             dropout=dropout_rate,
        #                             edge_dim=edge_dim,
        #                             beta=True) 

        # self.transf2 = Linear(embedding_size, embedding_size)
        # self.bn2 = BatchNorm1d(embedding_size)
        # ---------------------------------------------

        # Linear layers
        # TODO: only linear layers should be changed. either removing them or changing the last linear layer
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2)) # dafault: 128, 64
        self.linear3 = Linear(int(dense_neurons/2), int(dense_neurons/2))

    def forward(self, x, edge_attr, edge_index, batch_index):
        torch.autograd.set_detect_anomaly(True)
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        # x = torch.relu(self.transf1(x))
        x = self.leakyrelu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            # x = torch.relu(self.transf_layers[i](x))
            x = self.leakyrelu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
               x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                   x, edge_index, edge_attr, batch_index
                   )
               # Add current representation
               global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        # # test ------------------------------------
        # x = self.conv1(x, edge_index, edge_attr)
        # # x = torch.relu(self.transf1(x))
        # x = F.elu(x)
        # # x = self.bn1(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv2(x, edge_index, edge_attr)
        # x = F.elu(x)
        # # x = self.bn1(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = torch.relu(self.transf2(x))
        # # x = self.bn2(x)
        # x = gmp(x, batch_index)
        # # -----------------------------------------
    

        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x