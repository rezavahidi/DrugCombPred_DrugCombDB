
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
