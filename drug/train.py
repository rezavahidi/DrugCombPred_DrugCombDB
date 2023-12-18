import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import GCN
from datasets import DDInteractionDataset



if __name__ == '__main__':
    ddiDataset = DDInteractionDataset
    model = GCN(ddiDataset.num_features, ddiDataset.num_features // 2)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training on CPU
    n_epochs = 6
    for epoch in range(1, n_epochs):
        optimizer.zero_grad()
        out = model(ddiDataset.get().x, ddiDataset.get().edge_index)
        # TODO: MSELoss of the synergy scores
        loss = F.cross_entropy(out, data.y)
        
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss}")