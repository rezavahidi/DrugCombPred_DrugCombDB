import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
from torch_geometric.data import DataLoader
import time

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
        self.transformerGNN = TransformerGNN(feature_size=self.moleculeDataset[0].x.shape[1], model_params=drug_model_params).cuda(self.gpu_id)
        self.gcn = GCN(self.moleculeDataset[0].x.shape[1], self.moleculeDataset[0].x.shape[1] * 2).cuda(self.gpu_id)

    def forward(self, drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat):

        drug1_idx = torch.flatten(drug1_idx)
        drug2_idx = torch.flatten(drug2_idx)
        drug1_idx = drug1_idx.type(torch.long)
        drug2_idx = drug2_idx.type(torch.long)
        
        drug_index = torch.unique(torch.cat((drug1_idx,drug2_idx)))

        # start = time.time()

        subset = torch.utils.data.Subset(self.moleculeDataset, drug_index)

        feat_d = int(60)
        all_drug_feat = torch.empty((0, feat_d), dtype=torch.float32)
        # GCN
        # all_drug_feat = torch.empty((0, self.moleculeDataset[0].x.shape[1] * 2), dtype=torch.float32)
        if self.gpu_id is not None:
            all_drug_feat = all_drug_feat.cuda(self.gpu_id)
        train_loader = DataLoader(subset, batch_size=DRUG_MODEL_HYPERPARAMETERS["batch_size"], shuffle=True)
        for _, batch in enumerate(train_loader):
            if self.gpu_id is not None:
                batch = batch.cuda(self.gpu_id)
 
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
        

            # Passing the node features and the connection info
            # GCN:
            drug_feat = self.gcn(x.float(), 
                                    edge_index)

            # GAT:

            # drug_feat = self.transformerGNN(x.float(), 
            #                         edge_attr.float(),
            #                         edge_index, 
            #                         batch.batch)
            all_drug_feat = torch.cat((all_drug_feat, drug_feat), 0)
            
        # print("molecule loop end time:", time.time() - start)
        value_to_index = {value.item(): index for index, value in enumerate(drug_index)}
        drug1_idx = torch.tensor([value_to_index[value.item()] for value in drug1_idx])
        drug1_feat = all_drug_feat[drug1_idx]
        drug2_idx = torch.tensor([value_to_index[value.item()] for value in drug2_idx])
        drug2_feat = all_drug_feat[drug2_idx]

        feat = torch.cat([drug1_feat, drug2_feat, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat], 1)
        return feat


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, gpu_id=None):
        super(MLP, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=8)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, 1)

        self.connector = Connector(gpu_id)
    
    def forward(self, drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat):
        feat = self.connector(drug1_idx, drug2_idx, drug1_fp, drug2_fp, drug1_dti, drug2_dti, cell_feat)
        
        # Reshape feat for attention layer, assuming feat is [batch_size, seq_len, input_size]
        # feat = feat.unsqueeze(0)  # Add a dummy batch dimension if necessary
        # attn_output, attn_output_weights = self.attention(feat, feat, feat)
        
        # # Reshape back if needed
        # attn_output = attn_output.squeeze(0)
        
        out = self.linear1(feat)
        out = self.relu(out)
        out = self.batch_norm1(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.batch_norm2(out)
        out = self.output(out)
        return out


