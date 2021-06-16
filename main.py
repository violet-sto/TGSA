import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import load_data
from utils import EarlyStopping, set_random_seed
from utils import train, validate
from models.TGDRP import TGDRP

import argparse
import fitlog


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda:7',
                        help='device')
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
    parser.add_argument('--edge', type=str, default='PPI_0.95', help='threshold for cell line graph')
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

    drug_dict = np.load('./data/feature/drug_feature_graph.npy', allow_pickle=True).item()
    cell_dict = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_all.npy',
                        allow_pickle=True).item()
    edge_index = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_{}.npy'.format(args.edge))
    IC = pd.read_csv('./data/PANCANCER_IC_82833_580_170.csv')

    train_loader, val_loader, test_loader = load_data(IC, drug_dict, cell_dict, edge_index, 'My_model', args)
    print(len(IC), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print('mean degree:{}'.format(len(edge_index[0]) / 706))

    # model = My_model_motify_now(args).to(args.device)
    model = TGDRP(args).to(args.device)
    # model = My_model(args).to(args.device)
    model.load_state_dict(torch.load('./TGDRP_weights/TGDRP.pth', map_location=args.device))

    if args.mode == 'train':
        if args.pretrain and args.weight_path != '':
            model.GNN_drug.load_state_dict(torch.load('./model_pretrain/{}.pth'.format(args.weight_path)))

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_folder = os.path.join(os.getcwd(), "logs", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)

        stopper = EarlyStopping(mode='lower', patience=args.patience)
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_loss = train(model, train_loader, criterion, opt, args.device)
            fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)

            print('Evaluating...')
            rmse, _, _, _ = validate(model, val_loader, args.device)
            print("Validation rmse:{}".format(rmse))
            fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

            early_stop = stopper.step(rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)

        train_rmse, train_MAE, train_r2, train_r = validate(model, train_loader, args.device)
        val_rmse, val_MAE, val_r2, val_r = validate(model, val_loader, args.device)
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
        print('Train reslut: rmse:{} r2:{} r:{}'.format(train_rmse, train_r2, train_r))
        print('Val reslut: rmse:{} r2:{} r:{}'.format(val_rmse, val_r2, val_r))
        print('Test reslut: rmse:{} r2:{} r:{}'.format(test_rmse, test_r2, test_r))

        fitlog.add_best_metric(
            {'epoch': epoch - args.patience,
             "train": {'RMSE': train_rmse, 'MAE': train_MAE, 'pearson': train_r, "R2": train_r2},
             "valid": {'RMSE': stopper.best_score, 'MAE': val_MAE, 'pearson': val_r, 'R2': val_r2},
             "test": {'RMSE': test_rmse, 'MAE': test_MAE, 'pearson': test_r, 'R2': test_r2}})

    elif args.mode == 'test':
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
        print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 3), round(test_MAE, 3),
                                                             round(test_r2, 3), round(test_r, 3)))

if __name__ == "__main__":
    main()
