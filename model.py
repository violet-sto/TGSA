import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        self.drug_emb = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Conv1d(in_channels=40, out_channels=80, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Conv1d(in_channels=80, out_channels=60, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )
        self.cell_emb = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=40, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=40, out_channels=80, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Conv1d(in_channels=80, out_channels=60, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
        )

        self.regression = nn.Sequential(
            nn.Linear(2100, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1)
        )


    def forward(self, drug, cell):
        cell = cell.unsqueeze(-2)
        drug = self.drug_emb(drug)
        cell = self.cell_emb(cell)

        x = torch.cat([drug, cell], -1)
        x = x.reshape((x.shape[0], -1))
        x = self.regression(x)
        # x = torch.sigmoid(x)

        return x
