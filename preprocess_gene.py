import numpy as np
import pandas as pd
import os
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_genes_graph(genes_path, save_path, method='pearson', thresh=0.5, p_value=False):
    """
    根据表达量计算基因相关系数，确定邻接矩阵
    :param genes_exp_path:
    :return: 根据基因表达量构图
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    genes_exp_df = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)
    # 计算相关矩阵
    genes_exp_corr = genes_exp_df.corr(method=method)
    genes_exp_corr = genes_exp_corr.apply(lambda x: abs(x))
    n = genes_exp_df.shape[0]

    # 根据相关系数确定邻接矩阵(未考虑p值，因为p=0.05的时候，图太密集）
    if p_value == True:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)

    adj = np.where(genes_exp_corr > thresh, 1, 0)
    adj = adj - np.eye(genes_exp_corr.shape[0], dtype=np.int)
    edge_index = np.nonzero(adj)
    np.save(os.path.join(save_path, 'edge_index_{}_{}.npy').format(method, thresh), edge_index)

    return n, edge_index


def save_cell_graph(genes_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)
    cn = pd.read_csv(os.path.join(genes_path, 'cn.csv'), index_col=0)
    mu = pd.read_csv(os.path.join(genes_path, 'mu.csv'), index_col=0)
    print('缺失值：{}，{}，{}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum()))

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
    print('缺失值：{}，{}，{}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum()))

    cell_dict = {}
    for i in cell_names:
        cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = [np.array(exp.loc[i], dtype=np.float32), np.array(cn.loc[i], dtype=np.float32),
        #                 np.array(mu.loc[i], dtype=np.float32)]
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([cn.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([mu.loc[i]], dtype=torch.float).T)

    np.save(os.path.join(save_path, 'cell_feature_all.npy'), cell_dict)
    print("finish saving cell mut data!")


if __name__ == '__main__':
    # 注意基因个数，两个函数和两个参数都要改

    gene_path = './data/CellLines_DepMap/CCLE_605_18281/census_706'
    save_path = './data/CellLines_DepMap/CCLE_605_18281/census_706/'
    # get_genes_graph(gene_path, save_path, thresh=0.35, method='spearman')
    # get_genes_graph(gene_path, save_path, thresh=0.45, method='spearman')
    # get_genes_graph(gene_path, save_path, thresh=0.55, method='pearson')

    save_cell_graph(gene_path, save_path)
