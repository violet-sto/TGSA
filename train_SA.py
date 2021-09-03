import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from models.SA import SA
from utils import *
import argparse
import fitlog
import pickle
import pandas as pd


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:9',
                        help='device')
    parser.add_argument('--knn', type=int, default=5,
                        help='k-nearest-neighbour')
    parser.add_argument('--layer_drug', type=int, default=3, help='layer for drug')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug')
    parser.add_argument('--layer', type=int, default=3, help='number of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size (default: 512)')
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
    parser.add_argument('--edge', type=float, default='0.95', help='edge for gene graph')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--pretrain', type=int, default=1, help='pretrain(0 or 1)')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')

    return parser.parse_args()    


def main():
    args = arg_parse()
    log_folder = os.path.join(os.getcwd(), "logs", "SA")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    fitlog.set_log_dir(log_folder)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    set_random_seed(args.seed)
    train_loader, val_loader, test_loader = load_data_SA(args)
    test_rmse_list, test_MAE_list, test_r2_list, test_r_list = [], [], [], []
    for seed in range(42, 52):
        args.seed = seed
        set_random_seed(args.seed)
        drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, parameter = load_graph_data_SA(args)
        model = SA(drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, args).to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()
        model.regression = parameter['regression']
        model.drug_emb = parameter['drug_emb']
        model.cell_emb = parameter['cell_emb']
        stopper = EarlyStopping(mode='lower', patience=args.patience)
        print("train_num: {},  val_num: {}, test_num: {}".format(len(train_loader.dataset), len(val_loader.dataset),
                                                                    len(test_loader.dataset)))
        print("seed: {}".format(seed))
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_SA(train_loader, model, criterion, opt, args)
            print('Evaluating...')
            rmse, _, _, _ = validate_SA(val_loader, model, args)
            print("Validation rmse:{}".format(rmse))
            early_stop = stopper.step(rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)
        os.remove(stopper.filename)
        torch.save({'model_state_dict': model.state_dict()}, './weights/SA_pre_{}_{}_knn.pth'.format(seed, args.knn))

        test_rmse, test_MAE, test_r2, test_r = validate_SA(test_loader, model, args)
        print('Test reslut: rmse:{} MAE:{} r2:{} r:{}'.format(test_rmse, test_MAE, test_r2, test_r))
        fitlog.add_metric({'rmse_aug': test_rmse, 'MAE_aug': test_MAE, 'r_aug': test_r, "r2_aug": test_r2}, step=seed)

        test_rmse_list.append(test_rmse.item())
        test_MAE_list.append(test_MAE.item())
        test_r_list.append(test_r)
        test_r2_list.append(test_r2.item())

    fitlog.add_best_metric({
        'rmse': {'mean': np.mean(test_rmse_list).round(3), 'std': np.std(test_rmse_list, ddof=1).round(3)},
        'MAE': {'mean': np.mean(test_MAE_list).round(3), 'std': np.std(test_MAE_list, ddof=1).round(3)},
        'r2': {'mean': np.mean(test_r2_list).round(3), 'std': np.std(test_r2_list, ddof=1).round(3)},
        'r': {'mean': np.mean(test_r_list).round(3), 'std': np.std(test_r_list, ddof=1).round(3)}
    })
    fitlog.finish()

if __name__ == '__main__':
    main()


        