import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils import MyDataset
from utils import EarlyStopping, set_random_seed
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from model import TCNN
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def train(model, loader, criterion, opt, device):
    model.train()

    for idx, data in enumerate(tqdm(loader, desc='Iteration')):
        drug, cell, label = data
        drug, cell, label = drug.to(device).float(), cell.to(device).float(), label.to(device)
        output = model(drug, cell)
        loss = criterion(output, label.view(-1, 1).float())
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Train Loss:{}'.format(loss))
    return loss


def validate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):
            drug, cell, label = data
            drug, cell, label = drug.to(device).float(), cell.to(device).float(), label.to(device)
            output = model(drug, cell)
            total_loss += F.mse_loss(output, label.view(-1, 1).float(), reduction='sum')
            y_true.append(label.view(-1, 1))
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    rmse = torch.sqrt(total_loss / len(loader.dataset))
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    r = pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())

    return rmse, r2, r


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping')

    return parser.parse_args()


def main():
    args = arg_parse()
    set_random_seed(args.seed)

    drug_dict = np.load('./data/drug_feature.npy', allow_pickle=True).item()
    cell_dict = np.load('./data/cell_feature.npy', allow_pickle=True).item()
    IC = pd.read_csv('./data/PANCANCER_IC.csv')
    IC = IC[IC['Drug name'].isin(list(drug_dict.keys())) & IC['Cell line name'].isin(list(cell_dict.keys()))]
    IC['IC50'] = 1 / (1 + pow(np.exp(IC['IC50']), -0.1))
    IC.index = range(len(IC))
    Dataset = MyDataset(drug_dict, cell_dict, IC)

    train_size = int(0.8 * len(Dataset))
    val_size = int(0.1 * len(Dataset))
    test_size = len(Dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(Dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TCNN().to(args.device)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    stopper = EarlyStopping(mode='lower', patience=args.patience)
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_loss = train(model, train_loader, criterion, opt, args.device)

        print('Evaluating...')
        rmse, _, _ = validate(model, val_loader, args.device)
        print("Validation rmse:{}".format(rmse))

        early_stop = stopper.step(rmse, model)
        if early_stop:
            break

    print('EarlyStopping! Finish training!')
    print('Testing...')
    stopper.load_checkpoint(model)

    train_rmse, train_r2, train_r = validate(model, train_loader, args.device)
    val_rmse, val_r2, val_r = validate(model, val_loader, args.device)
    test_rmse, test_r2, test_r = validate(model, test_loader, args.device)
    print('Train reslut: rmse:{} r2:{} r:{}'.format(train_rmse, train_r2, train_r))
    print('Val reslut: rmse:{} r2:{} r:{}'.format(val_rmse, val_r2, val_r))
    print('Test reslut: rmse:{} r2:{} r:{}'.format(test_rmse, test_r2, test_r))


if __name__ == "__main__":
    main()
