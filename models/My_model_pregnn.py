import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, \
    graclus, max_pool, max_pool_x
import torch_geometric.transforms as T
from torch_geometric.nn import JumpingKnowledge
import numpy as np
from chem.model import GNN_graphpred
from chem.my_model import my_GNN_graphpred, my_GNN


class My_model_pregnn(nn.Module):
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

        self.convs_cell = torch.nn.ModuleList()
        # self.JK = JumpingKnowledge('cat')
        self.cluster_predefine = np.load(
            './data/CellLines_DepMap/CCLE_580_18281/census_706/cluster_predefine_{}.npy'.format(args.edge),
            allow_pickle=True).item()
        self.cluster_predefine = {i: j.to(args.device) for i, j in self.cluster_predefine.items()}
        self.bns_cell = torch.nn.ModuleList()
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())

        self.drug_gnn = my_GNN(num_layer=self.layer_drug, dim_drug=self.dim_drug)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * 3, 128),
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

        x_drug = global_max_pool(self.drug_gnn(drug.x, drug.edge_index), drug.batch)
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
