import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
from torch_geometric.data import DataLoader

PROJ_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

sys.path.insert(0, PROJ_DIR)
from drug.models import GCN, TransformerGNN
from drug.datasets import DDInteractionDataset, MoleculeDataset
from model.utils import get_FP_by_negative_index
from config import DRUG_MODEL_HYPERPARAMETERS



class Connector(nn.Module):
    def __init__(self, gpu_id=None):
        self.gpu_id = gpu_id
        super(Connector, self).__init__()

        self.moleculeDataset = MoleculeDataset(root = "drug/data/")
        drug_model_params = DRUG_MODEL_HYPERPARAMETERS
        drug_model_params["model_edge_dim"] = self.moleculeDataset[0].edge_attr.shape[1]
        self.transformerGNN = TransformerGNN(feature_size=self.moleculeDataset[0].x.shape[1], model_params=drug_model_params)

    def forward(self, drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat):

        drug1_idx = torch.flatten(drug1_idx)
        drug2_idx = torch.flatten(drug2_idx)
        drug1_idx = drug1_idx.type(torch.long)
        drug2_idx = drug2_idx.type(torch.long)
        
        drug_index = torch.unique(torch.cat((drug1_idx,drug2_idx)))

        feat_d = int(DRUG_MODEL_HYPERPARAMETERS["model_dense_neurons"]/2)
        all_drug_feat = torch.empty((0, feat_d), dtype=torch.float32)
        train_loader = DataLoader(self.moleculeDataset, batch_size=DRUG_MODEL_HYPERPARAMETERS["batch_size"], shuffle=True)
        for _, batch in enumerate(train_loader):
            if self.gpu_id is not None:
                batch = batch.cuda(self.gpu_id)
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
        

            # Passing the node features and the connection info
            drug_feat = self.transformerGNN(x.float(), 
                                    edge_attr.float(),
                                    edge_index, 
                                    batch.batch)
            all_drug_feat = torch.cat((all_drug_feat, drug_feat), 0)
            
        drug1_feat = all_drug_feat[drug1_idx]
        drug2_feat = all_drug_feat[drug2_idx]

        feat = torch.cat([drug1_feat, drug2_feat, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat], 1)
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
    
    def forward(self, drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat): # prev input: self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor
        feat = self.connector(drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat)
        out = self.layers(feat)
        return out

