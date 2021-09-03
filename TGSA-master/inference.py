import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import load_data
from utils import EarlyStopping, set_random_seed, load_data
from utils import train, validate, gradient, inference
from preprocess_gene import get_STRING_graph, get_predefine_cluster
from models.TGDRP import TGDRP

import argparse
import fitlog
import pickle


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='device')
    parser.add_argument('--model', type=str, default='TGDRP', help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--layer_drug', type=int, default=3, help='layer for drug')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--edge', type=float, default=0.9, help='threshold for cell line graph')
    parser.add_argument('--setup', type=str, default='known', help='experimental setup')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='whether use pre-trained weights (0 for False, 1 for True')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    return parser.parse_args()


def main():
    args = arg_parse()
    set_random_seed(args.seed)
    genes_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    
    cell_dict = np.load(os.path.join(genes_path, 'cell_feature_all.npy'), allow_pickle=True).item()
    drug_dict = np.load('./data/Drugs/drug_feature_graph.npy', allow_pickle=True).item()
    args.num_feature = cell_dict['ACH-000001'].x.shape[1]
    edge_index = get_STRING_graph(genes_path, args.edge)
    cluster_predefine = get_predefine_cluster(edge_index, genes_path, args.edge, args.device)
    model = TGDRP(cluster_predefine, args).to(args.device)
    model.load_state_dict(torch.load('./model_final_0.883/TGDRP_pre.pth', map_location=args.device)['model_state_dict'])
    
    IC = pd.read_csv('./data/PANCANCER_IC_82833_580_170.csv')
    train_loader, val_loader, test_loader = load_data(IC, drug_dict, cell_dict, edge_index, args)
    test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
    print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 4), round(test_MAE, 4),
                                                             round(test_r2, 4), round(test_r, 4)))
    
    
    with open("./data/similarity_augment/dict/gene_name_dict", "rb") as f:
        gene_name = pickle.load(f)
        
    # drug_name = 'Afatinib'
    # cell_name = 'ACH-000878'
    # ic50, importance, indice = gradient(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
    # print(f"the first 10th important genes with respect to drug {drug_name} for cellline {cell_name}")
    # print("idx:")
    # print(gene_name[indice[0:10]].tolist())
    # print("importance values:")
    # print(importance[0:10].tolist())
    # print(f'ic50: {ic50.item()}')
    # inference(model, drug_dict, cell_dict, edge_index, "./IC50_final_version.xlsx", args)
    
    # writer = pd.ExcelWriter("./gradient_final_version.xlsx")
    # for name in ["train", "val", "test"]:
    #     table = pd.read_excel("./IC50_final_version.xlsx", sheet_name=name)
    #     table.sort_values("IC50", ascending=True, inplace=True)
    #     table = table[0:100]
    #     important_genes_list = []
    #     drug_name_list, cell_name_list = table["Drug name"].tolist(), table["DepMap_ID"].tolist()
    #     input_list = zip(drug_name_list, cell_name_list)
    #     for drug_name, cell_name in input_list:
    #         ic50, indice = gradient(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args)
    #         important_genes_list.append(','.join(gene_name[indice[0:20]].tolist()))
    #     table["important_genes"] = important_genes_list
    #     table.to_excel(writer, sheet_name=name, index=False)
    # writer.close()
        
        
        

if __name__ == "__main__":
    main()
