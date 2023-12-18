import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import random
import os



class DDInteractionDataset(Dataset):
    def __init__(self, root = "\\drug/data/", transform=None, pre_transform=None, pre_filter=None, gpu_id=None):
        self.gpu_id = gpu_id
        super(DDInteractionDataset, self).__init__(os.path.dirname(os.path.abspath(os.path.dirname( __file__ ))) + "/drug/data/", transform, pre_transform, pre_filter)


    @property
    def num_features(self):
        return self._num_features
    
    @num_features.setter
    def num_features(self, value):
        self._num_features = value

    @property
    def raw_file_names(self):
        return ['new_interaction.tsv']

    @property
    def processed_file_names(self):
        return ['ddi_processed.pt']
    
    @property
    def raw_dir(self):
        dir = osp.join(self.root, 'DDI/DrugBank/raw')
        return dir

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, 'DDI/DrugBank/' + name)

    def download(self):
        pass

    def find_drugBank_id(self, index):
        path = osp.join(self.root, 'DDI/DrugBank/raw/' + 'drug2id.tsv')
        drug2id_df = pd.read_csv(path, sep='\t')
        drugBankID = drug2id_df['DrugBank_id'][index]
        return drugBankID
    
    def generate_rand_fp(self):
        number = random.getrandbits(256)

        # Convert the number to binary
        binary_string = '{0:0256b}'.format(number)
        random_fp = [x for x in binary_string]
        random_fp = list(map(int, random_fp))
        return random_fp

    def read_node_features(self, num_nodes):
        drug_fp_path = osp.join(self.root, 'RDkit extracted/drug2FP.csv')
        drug_fp_df = pd.read_csv(drug_fp_path)

        node_features = list()
        node_ids = list()
        for i in range(num_nodes):
            drugbankid =  self.find_drugBank_id(i)
            fp = drug_fp_df.loc[drug_fp_df['DrugBank_id'] == drugbankid]
            if fp.empty:
                fp = self.generate_rand_fp()
            else:
                fp = list(fp.to_numpy()[0,1:])

            node_features.append(fp)
            node_ids.append(drugbankid)

        #add synergy file drugs to end of the graph

        drug_fp_synergy_path = osp.join(self.root, 'drug2FP_synergy.csv')
        drug_fp_synergy_df = pd.read_csv(drug_fp_synergy_path)
        for index, row in drug_fp_synergy_df.iterrows():
            node_ids.append(row[0])
            node_features.append(list(row.to_numpy()[1:]))

        self.num_features = len(node_features[0])

        return node_ids, node_features

    def process(self):
        path = osp.join(self.raw_dir, self.raw_file_names[0])
        ddi = pd.read_csv(path , sep='\t')
        edge_index = torch.tensor([ddi['drug1_idx'],ddi['drug2_idx']], dtype=torch.long)
        num_nodes = ddi['drug1_idx'].max() + 1
        node_ids, node_features = self.read_node_features(num_nodes)
        node_features = torch.tensor(node_features, dtype=torch.int)
        print("node features nrow and ncol: ",len(node_features),len(node_features[0]))

        # ---------------------------------------------------------------
        data = Data(x = node_features, edge_index = edge_index)
 
        if self.gpu_id is not None:
            data = data.cuda(self.gpu_id)

        if self.pre_filter is not None and not self.pre_filter(data):
            pass

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, osp.join(self.processed_dir, 'ddi_graph_dataset.pt'))


    def len(self):
        return len(self.processed_file_names)

    def get(self):
        data = torch.load(osp.join(self.processed_dir, 'ddi_graph_dataset.pt'))
        return data

# run for checking
# ddiDataset = DDInteractionDataset(root = "drug/data/")
# print(ddiDataset.get().edge_index.t())
# print(ddiDataset.get().x)
# print(ddiDataset.num_features)