import numpy as np
import torch

import random

from torch.utils.data import Dataset
from .utils import read_map, get_index_by_name, read_files, get_DTI_data, get_FP_data

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class FastSynergyDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, cell_feat_file, synergy_score_file, dti_feat_file, fp_feat_file, use_folds,
                 train=True):
        self.drug2id = read_map(drug2id_file, keep_str = True)
        self.cell2id = read_map(cell2id_file)
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        self.raw_samples = []
        self.train = train
        dti = get_DTI_data(dti_feat_file)
        fp = get_FP_data(fp_feat_file)
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        files_dict = read_files()
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                #fold = random.randint(0, 4)
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if (train and (int(self.drug2id[drug1]) != use_folds and int(self.drug2id[drug2]) != use_folds)) or ((not train) and (int(self.drug2id[drug1]) == use_folds or int(self.drug2id[drug2]) == use_folds)):
                        drug1_id = get_index_by_name(drug1,files_dict)
                        drug2_id = get_index_by_name(drug2,files_dict)
                        sample = [
                            # TODO: specify drug_feat 
                            # drug1_feat + drug2_feat + cell_feat + score
                            torch.IntTensor([drug1_id]),
                            torch.IntTensor([drug2_id]),
                            fp[drug1],
                            fp[drug2],
                            dti[drug1],
                            dti[drug2],
                            torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                            torch.FloatTensor([float(score)]),
                        ]
                        # print(sample)
                        self.samples.append(sample)
                        raw_sample = [drug1, drug2, self.cell2id[cellname], score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            sample = [
                                torch.IntTensor([drug2_id]),
                                torch.IntTensor([drug1_id]),
                                fp[drug2],
                                fp[drug1],
                                dti[drug2],
                                dti[drug1],
                                torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                                torch.FloatTensor([float(score)]),
                            ]
                            self.samples.append(sample)
                            raw_sample = [drug2, drug1, self.cell2id[cellname], score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        # print('-----------------------------')
        # print(self.samples)
        # print('-----------------')
        # print(self.samples[0])
        # print(self.samples[i][0] and self.samples[i][1] for i in indices)
        i1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        i2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        d1 = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        d2 = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        t1 = torch.cat([torch.unsqueeze(self.samples[i][4], 0) for i in indices], dim=0)
        t2 = torch.cat([torch.unsqueeze(self.samples[i][5], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][6], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][7], 0) for i in indices], dim=0)
        return i1, i2, d1, d2, t1, t2, c, y
