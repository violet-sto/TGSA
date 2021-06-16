import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_genes_graph(genes_path, save_path, method='pearson', thresh=0.95, p_value=False):
    """
    determining adjaceny matrix based on correlation
    :param genes_exp_path:
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    genes_exp_df = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)

    # calculate correlation matrix
    genes_exp_corr = genes_exp_df.corr(method=method)
    genes_exp_corr = genes_exp_corr.apply(lambda x: abs(x))
    n = genes_exp_df.shape[0]

    # binarize
    if p_value == True:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)

    adj = np.where(genes_exp_corr > thresh, 1, 0)
    adj = adj - np.eye(genes_exp_corr.shape[0], dtype=np.int)
    edge_index = np.nonzero(adj)
    np.save(os.path.join(save_path, 'edge_index_{}_{}.npy').format(method, thresh), edge_index)

    return n, edge_index



def ensp_to_hugo_map():
    with open('./data/9606.protein.info.v11.0.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}

    return ensp_map

def hugo_to_ncbi_map():
    with open('./data/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        hugo_map = {row[0]: int(row[1]) for row in csv_reader if row[1] != ""}

    return hugo_map

def STRING_graph(exp_path, thresh=0.95):

    # gene_list
    exp = pd.read_csv(exp_path, index_col=0)
    gene_list = exp.columns.to_list()
    gene_list = [int(gene[1:-1]) for gene in gene_list]

    # load STRING
    ensp_map = ensp_to_hugo_map()
    hugo_map = hugo_to_ncbi_map()
    edges = pd.read_csv('./data/9606.protein.links.detailed.v11.0.txt', sep=' ')

    # edge_index
    selected_edges = edges['combined_score'] > (thresh * 1000)
    edge_list = edges[selected_edges][["protein1", "protein2"]].values.tolist()

    edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in edge_list if
                 edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]

    edge_list = [[hugo_map[edge[0]], hugo_map[edge[1]]] for edge in edge_list if
                 edge[0] in hugo_map.keys() and edge[1] in hugo_map.keys()]
    edge_index = []
    for i in edge_list:
        if (i[0] in gene_list) & (i[1] in gene_list):
            edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
            edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
    edge_index = list(set(edge_index))
    edge_index = np.array(edge_index, dtype=np.int64).T

    # 保存edge_index
    print(len(gene_list))
    print(thresh, len(edge_index[0]) / len(gene_list))
    np.save(os.path.join('./data/CellLines_DepMap/CCLE_580_18281/census_706/', 'edge_index_PPI_{}.npy'.format(thresh)),
            edge_index)



def save_cell_graph(genes_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)
    cn = pd.read_csv(os.path.join(genes_path, 'cn.csv'), index_col=0)
    mu = pd.read_csv(os.path.join(genes_path, 'mu.csv'), index_col=0)
    print('Miss values：{}，{}，{}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum()))

    index = exp.index
    columns = exp.columns

    scaler = StandardScaler()
    exp = scaler.fit_transform(exp)
    # cn = scaler.fit_transform(cn)

    imp_mean = SimpleImputer()
    exp = imp_mean.fit_transform(exp)

    exp = pd.DataFrame(exp, index=index, columns=columns)
    cn = pd.DataFrame(cn, index=index, columns=columns)
    mu = pd.DataFrame(mu, index=index, columns=columns)
    cell_names = exp.index
    print('Miss values：{}，{}，{}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum()))

    cell_dict = {}
    for i in cell_names:
        cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = [np.array(exp.loc[i], dtype=np.float32), np.array(cn.loc[i], dtype=np.float32),
        #                 np.array(mu.loc[i], dtype=np.float32)]

    np.save(os.path.join(save_path, 'cell_feature_all.npy'), cell_dict)
    print("finish saving cell mut data!")


if __name__ == '__main__':

    gene_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    save_path = './data/CellLines_DepMap/CCLE_580_18281/census_706/'

    save_cell_graph(gene_path, save_path)
