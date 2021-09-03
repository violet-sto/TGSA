import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SA(nn.Module):
    def __init__(self, drug_nodes, cell_nodes, drug_edges, cell_edges, args):
        super(SA, self).__init__()
        self.drug_nodes_feature = drug_nodes.to(args.device)
        self.drug_edges = drug_edges.to(args.device)
        self.cell_nodes_feature = cell_nodes.to(args.device)
        self.cell_edges = cell_edges.to(args.device)
        self.dropout = nn.Dropout(args.dropout_ratio)
        self.dropout_ratio = args.dropout_ratio
        self.dim_cell = cell_nodes.size(1)
        self.dim_drug = drug_nodes.size(1)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio)
        )
        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio)
        )
        self.drug_conv = SAGEConv(self.dim_drug, 256)
        self.cell_conv_1 = SAGEConv(self.dim_cell, 1024)
        self.cell_conv_2 = SAGEConv(1024, 256)
        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512, 1)
        )

    def forward(self, drug, cell):
        drug_id = drug.long()
        cell_id = cell.long()
        drug_x = self.drug_nodes_feature
        cell_x = self.cell_nodes_feature
        drug_x = self.drug_emb(drug_x)
        # drug_x = self.dropout(F.relu(self.drug_conv(drug_x, self.drug_edges)))
        cell_x = self.dropout(F.relu(self.cell_conv_1(cell_x, self.cell_edges)))
        cell_x = self.dropout(F.relu(self.cell_conv_2(cell_x, self.cell_edges)))
        # cell_x = self.cell_emb(cell_x)
        drug_x = drug_x.squeeze()
        cell_x = cell_x.squeeze()
        x = torch.cat([drug_x[drug_id], cell_x[cell_id]], -1)
        x = self.regression(x)
        return x