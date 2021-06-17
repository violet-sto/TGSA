import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from models.TGDRP import TGDRP, SA
from utils import *
import argparse
import fitlog
import pickle
import pandas as pd



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--knn', type=int, default=5,
                        help='k-nearest-neighbour')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--edge', type=str, default='PPI_0.9', help='edge for gene graph')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')

    return parser.parse_args()    


def main():
    args = arg_parse()
    set_random_seed(args.seed)
    train_loader, val_loader, test_loader = load_data_SA(args)
    drug_nodes_data, cell_nodes_data, drug_edges, cell_edges = load_graph_data_SA(args)
    model = SA(drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, args).to(args.device)
    if args.mode == "train":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()
        parameter = torch.load("similarity_augment/parameter/parameter.pth", map_location=args.device)
        model.regression = parameter['regression']
        model.drug_emb = parameter['drug_emb']
        model.drug_conv.load_state_dict(parameter['drug_conv'])
        log_folder = os.path.join(os.getcwd(), "logs", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)
        set_random_seed(args.seed)
        stopper = EarlyStopping(mode='lower', patience=args.patience)
        print("train_num: {},  val_num: {}, test_num: {}".format(len(train_loader.dataset), len(val_loader.dataset),
                                                                    len(test_loader.dataset)))
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train(train_loader, model, criterion, opt, args)
            print('Evaluating...')
            rmse, _, _, _ = validate(val_loader, model, args)
            print("Validation rmse:{}".format(rmse))
            early_stop = stopper.step(rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)

        train_rmse, train_MAE, train_r2, train_r = validate(model, train_loader, args)
        val_rmse, val_MAE, val_r2, val_r = validate(model, val_loader, args)
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args)
        print('Train reslut: rmse:{} r2:{} r:{}'.format(train_rmse, train_r2, train_r))
        print('Val reslut: rmse:{} r2:{} r:{}'.format(val_rmse, val_r2, val_r))
        print('Test reslut: rmse:{} r2:{} r:{}'.format(test_rmse, test_r2, test_r))

        fitlog.add_best_metric(
            {'epoch': epoch - args.patience,
            "train": {'RMSE': train_rmse, 'MAE': train_MAE, 'pearson': train_r, "R2": train_r2},
            "valid": {'RMSE': stopper.best_score, 'MAE': val_MAE, 'pearson': val_r, 'R2': val_r2},
            "test": {'RMSE': test_rmse, 'MAE': test_MAE, 'pearson': test_r, 'R2': test_r2}})
        fitlog.finish()

    elif args.mode == "test":
        model.load_state_dict(torch.load('./weights/SA.pth', map_location=args.device))
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args)
        print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 3), round(test_MAE, 3),
                                                             round(test_r2, 3), round(test_r, 3)))


if __name__ == '__main__':
    main()

