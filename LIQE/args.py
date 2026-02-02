import argparse
import torch


def Configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sv_path', dest='sv_path', type=str, default='/data/hlz/models/LIQE/WCKD')
    parser.add_argument('--lr', dest='lr', type=float, default=5e-6,
                        help='Learning rate')
    parser.add_argument('--train_patch', dest='train_patch', type=int, default=3,
                        help='Train Patch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=80,
                        help='Epochs for training')
    parser.add_argument('--seed', dest='seed', type=int, default=2024,
                        help='for reproducing the results')
    parser.add_argument('--gpunum', dest='gpunum', type=str, default='2',
                        help='the id for the gpu that will be used')
    parser.add_argument('--cov', dest='cov', type=bool, default=True,
                        help='whether covariance constraint exists or not')
    parser.add_argument('--fusion', dest='fusion', type=bool, default=True,
                        help='perform feature fusion or not')
    return parser.parse_args()