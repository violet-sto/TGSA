import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, JumpingKnowledge, GATConv
from torch_geometric.nn import global_max_pool, max_pool, global_add_pool, global_mean_pool
from models.GNN_drug import GNN_drug
import torch_geometric.transforms as T


class My_model_drug(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = 128
        self.layer_drug = 3
        self.dim_drug = 128
        self.layer_cell = 2
        self.dim_cell = 16
        self.dropout_ratio = 0.2
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_PPI_0.95.npy',
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
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


class My_model_drug_pre(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = 128
        self.layer_drug = 3
        self.dim_drug = 128
        self.layer_cell = 3
        self.dim_cell = 4
        self.dropout_ratio = 0.2
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_PPI_0.9.npy',
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
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


class My_model_cell(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = 128
        self.layer_drug = 3
        self.dim_drug = 128
        self.layer_cell = 2
        self.dim_cell = 8
        self.dropout_ratio = 0.2
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_PPI_0.9.npy',
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
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


class My_model_motify_now(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = 128
        self.layer_drug = 3
        self.dim_drug = 128
        self.layer_cell = 2
        self.dim_cell = 8
        self.dropout_ratio = 0.2
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_PPI_0.95.npy',
            allow_pickle=True).item()

        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
                # conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
                # conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
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


class My_model_motify(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
                # conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
                # conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
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


class My_model_motify_2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
                # conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
                # conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        for i in range(self.layer_cell):
            cell.x = self.convs_cell[i](cell.x, cell.edge_index)
            num_node = int(cell.x.size(0) / cell.num_graphs)
            ## 输入预先计算的cluster
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)
            cell.x = F.relu(cell.x)

        x_cell = cell.x.reshape(-1, self.final_node * self.dim_cell)

        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x


class My_model_motify_prelu(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # cell graph branch
        for i in range(self.layer_cell):

            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
                # conv = SAGEConv(self.dim_cell, self.dim_cell)

            else:
                conv = GATConv(3, self.dim_cell)
                # conv = SAGEConv(3, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # False
            activation = nn.PReLU(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)
            self.activations.append(activation)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 384),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(384, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch

        node_representation = self.GNN_drug(x, edge_index)
        x_drug = global_max_pool(node_representation, batch)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        for i in range(self.layer_cell):
            cell.x = self.activations[i](self.convs_cell[i](cell.x, cell.edge_index))
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