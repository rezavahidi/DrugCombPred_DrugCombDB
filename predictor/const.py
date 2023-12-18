import os
# e:\Me\Master\BCB\Thesis\DrugCombModel
# PROJ_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

# E:\Me\Master\BCB\Thesis\DrugCombModel\DrugCombinationPredeiction
PROJ_DIR = os.path.dirname(os.path.abspath(os.path.dirname( __file__ )))
DRUG_DIR = os.path.join(PROJ_DIR, 'drug')
SUB_PROJ_DIR = os.path.join(PROJ_DIR, 'predictor')
DATA_DIR = os.path.join(SUB_PROJ_DIR, 'data')
DRUG_DATA_DIR = os.path.join(PROJ_DIR, 'drug', 'data')
CELL_DATA_DIR = os.path.join(PROJ_DIR, 'cell', 'data')
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SYNERGY_FILE = os.path.join(DATA_DIR, 'drug_combinations_small.tsv')

# DRUG_FEAT_FILE = os.path.join(DRUG_DATA_DIR, 'drug_feat.npy')
DRUGN2ID_FILE = os.path.join(DRUG_DATA_DIR, 'DDI\DrugBank\\raw\drug2id.tsv')
DRUGNAME_2_DRUGBANKID_FILE = os.path.join(DRUG_DATA_DIR, 'drugname2drugbankid.tsv')
CELL_FEAT_FILE = os.path.join(CELL_DATA_DIR, 'cell_features.npy')
CELL2ID_FILE = os.path.join(CELL_DATA_DIR, 'cell2id.tsv')
DTI_FEAT_FILE = os.path.join(DRUG_DATA_DIR, 'DTI/DTI_reduced.csv')
FP_FEAT_FILE = os.path.join(DRUG_DATA_DIR, 'FP/drug2FP_small.csv')
DRUG2ID_FILE = os.path.join(DRUG_DATA_DIR, 'drug2id.tsv')



