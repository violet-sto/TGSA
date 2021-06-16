import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, \
    graclus, max_pool, max_pool_x
import torch_geometric.transforms as T
from torch_geometric.nn import JumpingKnowledge
import numpy as np

class My_model_v3_graclus_2layer_dropout_SAGE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = 3
        self.dim_drug = 128
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.fc_cell = args.fc_cell
        self.out_cell = args.out_cell
        self.dropout_ratio = args.dropout_ratio

        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.convs_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.bns_cell = torch.nn.ModuleList()
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch

        for i in range(self.layer_drug):

            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                # conv = GCNConv(self.dim_cell, self.dim_cell)
                # conv = GATConv(self.dim_cell, self.dim_cell)
                conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                # conv = GCNConv(3, self.dim_cell)
                # conv = GATConv(3, self.dim_cell)
                conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, self.fc_cell),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_cell, self.out_cell),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(self.out_cell + 128, self.out_cell + 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.out_cell + 128, self.out_cell + 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.out_cell + 128, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)

        x_drug = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        # x, edge_index, batch = cell.x, cell.edge_index, cell.batch

        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            ## 输入预先计算的cluster
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        x_cell = cell.x.reshape(-1, self.final_node * self.dim_cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x


class My_model_v3_graclus(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.fc_cell = args.fc_cell
        self.out_cell = args.out_cell
        self.dropout_ratio = args.dropout_ratio

        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.convs_cell = torch.nn.ModuleList()
        self.JK = JumpingKnowledge('cat')
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.bns_cell = torch.nn.ModuleList()
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch

        for i in range(self.layer_drug):

            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

        self.drug_emb = nn.Sequential(
            # nn.Linear(self.dim_drug * 2, 128),
            # nn.Linear(self.dim_drug * 6, 128),
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                # conv = GCNConv(self.dim_cell, self.dim_cell)
                # conv = GATConv(self.dim_cell, self.dim_cell)
                conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                # conv = GCNConv(3, self.dim_cell)
                # conv = GATConv(3, self.dim_cell)
                conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, self.fc_cell),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_cell, self.out_cell),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(self.out_cell + 128, self.out_cell + 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.out_cell + 128, self.out_cell + 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.out_cell + 128, 1)
        )
        # self.x_embedding1 = torch.nn.Embedding(120, 128)
        # self.x_embedding2 = torch.nn.Embedding(3, 128)

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        # x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            # x_drug_list.append(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
            x_drug_list.append(global_max_pool(x, batch))

        # x_drug = global_add_pool(x, batch)
        # x_drug = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x_drug = self.JK(x_drug_list)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        # x, edge_index, batch = cell.x, cell.edge_index, cell.batch

        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            ## 输入预先计算的cluster
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        x_cell = cell.x.reshape(-1, self.final_node * self.dim_cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x


class My_model_v3_graclus_JK(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.fc_cell = args.fc_cell
        self.out_cell = args.out_cell
        self.dropout_ratio = args.dropout_ratio

        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        # self.JK = JumpingKnowledge('max')
        self.att = attention(args)
        self.convs_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.bns_cell = torch.nn.ModuleList()
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        for i in range(self.layer_drug):

            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                # conv = GCNConv(self.dim_cell, self.dim_cell)
                # conv = GATConv(self.dim_cell, self.dim_cell)
                conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                # conv = GCNConv(3, self.dim_cell)
                # conv = GATConv(3, self.dim_cell)
                conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, self.fc_cell),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_cell, self.out_cell),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(self.out_cell + 128, self.out_cell + 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.out_cell + 128, self.out_cell + 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.out_cell + 128, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1))
        # x_drug = global_add_pool(x, batch)
        # x_drug = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # x_drug = self.JK(x_drug_list)
        # x_drug = self.drug_emb(x_drug)

        # forward cell
        # x, edge_index, batch = cell.x, cell.edge_index, cell.batch

        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            ## 输入预先计算的cluster
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        x_cell = cell.x.reshape(-1, self.final_node * self.dim_cell)

        K = torch.stack(x_drug_list, dim=-2)

        x_drug = self.att(x_cell, K, K)

        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x
