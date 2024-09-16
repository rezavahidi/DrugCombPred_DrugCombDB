import os
import torch
import json
import random
import pandas as pd
import numpy as np
from const import SYNERGY_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR, DRUGNAME_2_DRUGBANKID_FILE, DRUG2ID_FILE

project_path = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
drug2FP_file = os.path.join(project_path, 'drug/data/drug2FP_synergy.csv')
#drug2FP_df = pd.read_csv(drug2FP_file)

def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma


def conf_inv(mu, sigma, n):
    delta = 2.776 * sigma / (n ** 0.5)  # 95%
    return mu - delta, mu + delta


def arg_min(lst):
    m = float('inf')
    idx = 0
    for i, v in enumerate(lst):
        if v < m:
            m = v
            idx = i
    return m, idx


def save_best_model(state_dict, model_dir: str, best_epoch: int, keep: int):
    save_to = os.path.join(model_dir, '{}.pkl'.format(best_epoch))
    torch.save(state_dict, save_to)
    model_files = [f for f in os.listdir(model_dir) if os.path.splitext(f)[-1] == '.pkl']
    epochs = [int(os.path.splitext(f)[0]) for f in model_files if str.isdigit(f[0])]
    outdated = sorted(epochs, reverse=True)[keep:]
    for n in outdated:
        os.remove(os.path.join(model_dir, '{}.pkl'.format(n)))


def find_best_model(model_dir: str):
    model_files = [f for f in os.listdir(model_dir) if os.path.splitext(f)[-1] == '.pkl']
    epochs = [int(os.path.splitext(f)[0]) for f in model_files if str.isdigit(f[0])]
    best_epoch = max(epochs)
    return os.path.join(model_dir, '{}.pkl'.format(best_epoch))


def save_args(args, save_to: str):
    args_dict = args.__dict__
    with open(save_to, 'w') as f:
        json.dump(args_dict, f, indent=2)


def read_map(map_file, keep_str = False):
    d = {}
    print(map_file)
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split("\t")
            if keep_str:
                d[k] = v
            else:
                d[k] = int(v)
    return d


def random_split_indices(n_samples, train_rate: float = None, test_rate: float = None):
    if train_rate is not None and (train_rate < 0 or train_rate > 1):
        raise ValueError("train rate should be in [0, 1], found {}".format(train_rate))
    elif test_rate is not None:
        if test_rate < 0 or test_rate > 1:
            raise ValueError("test rate should be in [0, 1], found {}".format(test_rate))
        train_rate = 1 - test_rate
    elif train_rate is None and test_rate is None:
        raise ValueError("Either train_rate or test_rate should be given.")
    evidence = list(range(n_samples))
    train_size = int(len(evidence) * train_rate)
    random.shuffle(evidence)
    train_indices = evidence[:train_size]
    test_indices = evidence[train_size:]
    return train_indices, test_indices

# --------------------------------------------------------------------- Our Part:

def read_files():
    #drug_name2drugbank_id_df = pd.read_csv(DRUGNAME_2_DRUGBANKID_FILE, sep='\s+')
    drug2id_df = pd.read_csv(DRUG2ID_FILE , sep='\t')

    return {'drug2id_df': drug2id_df}

def get_index_by_name(drug_name, files_dict = None):
    
    if files_dict == None:
        files_dict = read_files()

    drug2id_df = files_dict['drug2id_df']

    row = drug2id_df[drug2id_df['drug_name'] == drug_name]

    drug_index = row.id.item()

    return drug_index

def get_FP_by_negative_index(index):

    index = index.item()

    #array = np.array(list(drug2FP_df.iloc[-index])[1:])
    #return torch.tensor(array, dtype=torch.float32)


def get_DTI_data(dti_feat_file):
    df = pd.read_csv(dti_feat_file)
    DTI = {}
    for row in df.iterrows():
        DTI[row[1][0]] = torch.FloatTensor(row[1][1:])
    return DTI

def get_FP_data(df_feat_file):
    df = pd.read_csv(df_feat_file)
    DTI = {}
    for row in df.iterrows():
        DTI[row[1][0]] = torch.FloatTensor(row[1][1:])
    return DTI
    

