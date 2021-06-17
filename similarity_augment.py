import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import numpy as np
import pickle
from models.TGDRP import TGDRP
from utils import *
from rdkit import DataStructs,Chem
from rdkit.Chem import MACCSkeys,AllChem
from scipy.stats import pearsonr
import argparse

dir = 'similarity_augment/'
dict_dir = 'similarity_augment/dict/'
with open(dict_dir + "cell_id2idx_dict", 'rb') as f:
    cell_id2idx_dict = pickle.load(f)
with open(dict_dir + "drug_name_cell_id_ic50", 'rb') as f:
    drug_name_cell_id_ic50 = pickle.load(f)
with open(dict_dir + "drug_idx_cell_idx_ic50", 'rb') as f:
    drug_idx_cell_idx_ic50 = pickle.load(f)
with open(dict_dir + "drug_name2smiles_dict", 'rb') as f:
    drug_name2smiles_dict = pickle.load(f)
with open(dict_dir + "drug_idx2smiles_dict", 'rb') as f:
    drug_idx2smiles_dict = pickle.load(f)
with open(dict_dir + "drug_name2idx_dict", 'rb') as f:
    drug_name2idx_dict = pickle.load(f)
with open(dict_dir + "cell_idx2id_dict", 'rb') as f:
    cell_idx2id_dict = pickle.load(f)
with open(dict_dir + "drug_idx2name_dict", 'rb') as f:
    drug_idx2name_dict = pickle.load(f)
with open(dict_dir + "cell_feature_normalized", 'rb') as f:
    cell_feature_normalized = pickle.load(f)
with open(dict_dir + "cell_feature", 'rb') as f:
    cell_feature = pickle.load(f)
with open(dict_dir + "cell_sim_matrix", 'rb') as f:
    cell_sim_matrix = pickle.load(f)
with open(dict_dir + "drug_sim_matrix", 'rb') as f:
    drug_sim_matrix = pickle.load(f)
    
def computing_sim_matrix():
    drug_sim_matrix = np.zeros((len(drug_name2idx_dict), len(drug_name2idx_dict)))
    mi = [Chem.MolFromSmiles(drug_idx2smiles_dict[i]) for i in range(len(drug_name2idx_dict))]
    fps = [AllChem.GetMorganFingerprint(x, 4) for x in mi]
    for i in range(len(drug_name2idx_dict)):
        for j in range(len(drug_name2idx_dict)):
            if i != j:
                drug_sim_matrix[i][j] = DataStructs.DiceSimilarity(fps[i],fps[j])

    cell_sim_matrix = np.zeros((len(cell_id2idx_dict), len(cell_id2idx_dict)))
    for i in range(len(cell_id2idx_dict)):
        for j in range(len(cell_id2idx_dict)):
            if i != j:
                cell_sim_matrix[i][j], _ = pearsonr(cell_feature_normalized[i], cell_feature_normalized[j])
                if cell_sim_matrix[i][j] < 0:
                    cell_sim_matrix[i][j] = 0
    with open(dict_dir+ "cell_sim_matrix", 'wb') as f:
        pickle.dump(cell_sim_matrix, f)
    with open(dict_dir+ "drug_sim_matrix", 'wb') as f:
        pickle.dump(drug_sim_matrix, f)

def computing_knn(k):
    computing_sim_matrix()
    cell_sim_matrix_new = np.zeros_like(cell_sim_matrix)
    for u in range(len(cell_id2idx_dict)):
        v = cell_sim_matrix[u].argsort()[-6:-1]
        cell_sim_matrix_new[u][v] = cell_sim_matrix[u][v]
    drug_sim_matrix_new = np.zeros_like(drug_sim_matrix)
    for u in range(len(drug_name2idx_dict)):
        v = drug_sim_matrix[u].argsort()[-6:-1]
        drug_sim_matrix_new[u][v] = drug_sim_matrix[u][v]
    drug_edges = np.argwhere(drug_sim_matrix_new >  0)
    cell_edges = np.argwhere(cell_sim_matrix_new >  0)
    with open(dir + "edge/drug_cell_edges_{}_knn".format(k), 'wb') as f:
        pickle.dump((drug_edges, cell_edges), f)
    
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--knn', type=int, default=5,
                        help='knn')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--edge', type=str, default='PPI_0.9', help='edge for gene graph')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')

    return parser.parse_args(args=[])


def computing_parameters_SA():
    args = arg_parse()
    model = TGDRP(args).to(args.device)
    tgdrp = torch.load('weights/weights/TGDRP.pth', map_location=args.device)
    model.load_state_dict(tgdrp)
    drug_conv = {'lin_r.weight':tgdrp['drug_emb.0.weight'], 'lin_l.weight':torch.zeros_like(tgdrp['drug_emb.0.weight']), 'lin_l.bias':tgdrp['drug_emb.0.bias']}
    cell_conv_1 = {'lin_r.weight':tgdrp['cell_emb.0.weight'], 'lin_l.weight':torch.zeros_like(tgdrp['cell_emb.0.weight']), 'lin_l.bias':tgdrp['cell_emb.0.bias']}                       
    cell_conv_2 = {'lin_r.weight':tgdrp['cell_emb.3.weight'], 'lin_l.weight':torch.zeros_like(tgdrp['cell_emb.3.weight']), 'lin_l.bias':tgdrp['cell_emb.3.bias']}                             
    torch.save({'drug_emb':model.drug_emb, 'regression':model.regression, 'cell_emb':model.cell_emb, \
    'drug_conv': drug_conv, 'cell_conv_1':cell_conv_1, 'cell_conv_2': cell_conv_2}, "./data/similarity_augment/parameter/parameter.pth")
    
if __name__ == '__main__':
    computing_knn(5)
    computing_parameters_SA()