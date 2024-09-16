import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import random
import os
from tqdm import tqdm
from rdkit import Chem 
import deepchem as dc



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

class MoleculeDataset(Dataset):

    def __init__(self, root = "/drug/data/", test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        # self.atom_indices = self.get_atom_indices()
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. 
        """
        return ['DrugCombDB_drugs.csv']
    
    @property
    def raw_dir(self):
        dir = osp.join(self.root, 'Smiles/raw')
        return dir

    @property
    def processed_dir(self):
        return osp.join(self.root, 'Smiles/processed')

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        self.data = pd.read_csv(osp.join(os.path.dirname(os.path.abspath(__file__)), "data/Smiles/raw/DrugCombDB_drugs.csv")).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
    def download(self):
        pass

    def _custom_to_pyg_graph(self,graph_data):
        from torch_geometric.data import Data
        return Data(x=torch.from_numpy(graph_data.node_features).float(),
                    edge_index=torch.from_numpy(graph_data.edge_index).long(),
                    edge_attr=torch.from_numpy(graph_data.edge_features).float())


    def process(self):
        path = osp.join(self.raw_dir, self.raw_file_names[0])
        #self.data = pd.read_csv(path).reset_index()
        # deepchem ------------------------------
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        # deep chem end -------------------------
        self.data = pd.read_csv(osp.join(os.path.dirname(os.path.abspath(__file__)), "data/Smiles/raw/DrugCombDB_drugs.csv")).reset_index()
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            # mol_obj = Chem.MolFromSmiles(mol["smiles"])
            # # Get node features
            # node_feats = self.get_node_features(mol_obj)
            # # Get edge features
            # edge_feats = self.get_edge_features(mol_obj)
            # # Get adjacency info
            # edge_index = self.get_adjacency_info(mol_obj)

            # deepchem ----------------------------------
            mol_obj = Chem.MolFromSmiles(mol["smiles"])
            f = featurizer._featurize(mol_obj)
            # data = f.to_pyg_graph()
            data = self._custom_to_pyg_graph(f)
            data.id = mol['id']
            data.smiles = mol["smiles"]
            # deepchem end -------------------------------

            # Create data object
            # data = Data(x=node_feats, 
            #             edge_index=edge_index,
            #             edge_attr=edge_feats,
            #             id=mol['drugbank_id'],
            #             smiles=mol["smiles"]
            #             ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))

    # def get_node_features(self, mol):
    #     """ 
    #     This will return a matrix / 2d array of the shape
    #     [Number of Nodes, Node Feature size]
    #     """
    #     all_node_feats = []

    #     for atom in mol.GetAtoms():
    #         node_feats = []
    #         # Feature 1: Atomic number        
    #         node_feats.append(atom.GetAtomicNum())
    #         # Feature 2: Atom degree
    #         node_feats.append(atom.GetDegree())
    #         # Feature 3: Formal charge
    #         node_feats.append(atom.GetFormalCharge())
    #         # Feature 4: Hybridization
    #         node_feats.append(atom.GetHybridization())
    #         # Feature 5: Aromaticity
    #         node_feats.append(atom.GetIsAromatic())
    #         # Feature 6: Total Num Hs
    #         node_feats.append(atom.GetTotalNumHs())
    #         # Feature 7: Radical Electrons
    #         node_feats.append(atom.GetNumRadicalElectrons())
    #         # Feature 8: In Ring
    #         node_feats.append(atom.IsInRing())
    #         # Feature 9: Chirality
    #         node_feats.append(atom.GetChiralTag())

    #         # Append node features to matrix
    #         all_node_feats.append(node_feats)

    #     all_node_feats = np.asarray(all_node_feats)
    #     return torch.tensor(all_node_feats, dtype=torch.float)

    # def get_edge_features(self, mol):
    #     """ 
    #     This will return a matrix / 2d array of the shape
    #     [Number of edges, Edge Feature size]
    #     """
    #     all_edge_feats = []

    #     for bond in mol.GetBonds():
    #         edge_feats = []
    #         # Feature 1: Bond type (as double)
    #         edge_feats.append(bond.GetBondTypeAsDouble())
    #         # Feature 2: Rings
    #         edge_feats.append(bond.IsInRing())
    #         # Append node features to matrix (twice, per direction)
    #         all_edge_feats += [edge_feats, edge_feats]

    #     all_edge_feats = np.asarray(all_edge_feats)
    #     return torch.tensor(all_edge_feats, dtype=torch.float)

    # def get_adjacency_info(self, mol):
    #     """
    #     We could also use rdmolops.GetAdjacencyMatrix(mol)
    #     but we want to be sure that the order of the indices
    #     matches the order of the edge features
    #     """
    #     edge_indices = []
    #     for bond in mol.GetBonds():
    #         i = bond.GetBeginAtomIdx()
    #         j = bond.GetEndAtomIdx()
    #         edge_indices += [[i, j], [j, i]]

    #     edge_indices = torch.tensor(edge_indices)
    #     edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    #     return edge_indices

    def len(self):
        return self.data.shape[0]
    
    def get_by_idx(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt')) 
        return data

    def get(self, indices):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if isinstance(indices, int):
            idx = indices
            return self.get_by_idx(idx)
        else:
            data_list = []
            for idx in indices:
                data = self.get_by_idx(idx)
                data_list.append(data) 
            return data_list

    def get_atom_indices(self):
        atom_indices = {}
        counter = 0
        for i in range(616):
            atom_indices[i] = []
            atom_indices[i].append(counter)
            counter += len(self.get(i).x)
            atom_indices[i].append(counter)
        return atom_indices
            

# run for checking
# moleculeDataset = MoleculeDataset(root = "drug/data/")
# print(moleculeDataset.get(0).edge_index.t())
# print(moleculeDataset.get(0).x)
# print(moleculeDataset.get(0).id)