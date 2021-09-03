import torch
import torch.nn as nn
from models.GNN_drug import GNN_drug
from models.GNN_cell import GNN_cell


class TGDRP(nn.Module):
    def __init__(self, cluster_predefine, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.num_feature = args.num_feature
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        # cell graph branch
        self.GNN_cell = GNN_cell(self.num_feature, self.layer_cell, self.dim_cell, cluster_predefine)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 1)
        )

    def forward(self, drug, cell):
        # forward drug
        x_drug = self.GNN_drug(drug)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)

        return x


