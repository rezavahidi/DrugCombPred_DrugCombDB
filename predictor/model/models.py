import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 

PROJ_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

sys.path.insert(0, PROJ_DIR)
from drug.models import GCN
from drug.datasets import DDInteractionDataset
from model.utils import get_FP_by_negative_index



class Connector(nn.Module):
    def __init__(self, gpu_id=None):
        self.gpu_id = gpu_id
        super(Connector, self).__init__()
        #self.ddiDataset = DDInteractionDataset(gpu_id = gpu_id)
        #self.gcn = GCN(self.ddiDataset.num_features, self.ddiDataset.num_features * 2)
        
        #Cell line features
        # np.load('cell_feat.npy')

    def forward(self, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat):
        #x = self.ddiDataset.get().x
        #edge_index = self.ddiDataset.get().edge_index
        #x = self.gcn(x, edge_index)
        #drug1_idx = torch.flatten(drug1_idx)
        #drug2_idx = torch.flatten(drug2_idx)
        #drug1_idx = drug1_idx.type(torch.long)
        #drug2_idx = drug2_idx.type(torch.long)
        #drug1_feat = x[drug1_idx]
        #drug2_feat = x[drug2_idx]
        #drug1_feat = torch.empty((len(drug1_idx), len(x[0])))
        #drug2_feat = torch.empty((len(drug2_idx), len(x[0])))
        #for index, element in enumerate(drug1_idx):
        #    drug1_feat[index] = (x[element])
        #for index, element in enumerate(drug2_idx):
        #    drug2_feat[index] = (x[element])
        #if self.gpu_id is not None:
        #    drug1_feat = drug1_feat.cuda(self.gpu_id)
        #    drug2_feat = drug2_feat.cuda(self.gpu_id)
        #for i, x in enumerate(drug1_idx):
        #    if x < 0:
        #        drug1_feat[i] = get_FP_by_negative_index(x)
        #for i, x in enumerate(drug2_idx):
        #    if x < 0:
        #        drug2_feat[i] = get_FP_by_negative_index(x)
        feat = torch.cat([drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat], 1)
        return feat


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, gpu_id=None):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

        self.connector = Connector(gpu_id)
    
    def forward(self, drug1_idx, drug2_idx, drug1_dti, drug2_dti, cell_feat): # prev input: self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor
        feat = self.connector(drug1_idx, drug2_idx, drug1_dti, drug2_dti, cell_feat)
        out = self.layers(feat)
        return out


# other PRODeepSyn models have been deleted for now
