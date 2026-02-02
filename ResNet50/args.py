import argparse
import torch


def Configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/data/hlz/IQA_Database/CSIQ/',
                        help='ChallengeDB_release | CSIQ | databaserelease2 | kadid-10k | koniq-10k | TID2013 | BID/ImageDatabase')
    parser.add_argument('--dataset', dest='dataset', type=str, default='CSIQ',
                        help='Support datasets: LIVE|CSIQ|bid|CLIVE|Koniq-10k|Kadid-10k|TID2013')
    parser.add_argument('--sv_path', dest='sv_path', type=str, default='/data/hlz/models/ResNet50/WCKD/',)
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--train_patch', dest='train_patch', type=int, default=3,
                        help='Train Patch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=150,
                        help='Epochs for training')
    parser.add_argument('--seed', dest='seed', type=int, default=2024,
                        help='for reproducing the results')
    parser.add_argument('--gpunum', dest='gpunum', type=str, default='0',
                        help='the id for the gpu that will be used')
    parser.add_argument('--cov', dest='cov', type=bool, default=True,
                        help='whether covariance constraint exists or not')
    parser.add_argument('--fusion', dest='fusion', type=bool, default=True,
                        help='perform feature fusion or not')
    return parser.parse_args()