import os
import random
import json
import numpy as np
import torch
from args import Configs
import logging

from transformer import BaseVIT
from scatter import plot_scatter


def main(config, device):
    folder_path = {
        'live': config.datapath,
        'csiq': config.datapath,
        'tid2013': config.datapath,
        'kadid10k': config.datapath,
        'clive': config.datapath,
        'koniq': config.datapath,
        'fblive': config.datapath,
        'bid': config.datapath,
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013': list(range(0, 25)),
        'clive': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        'fblive': list(range(0, 39810)),
        'bid': list(range(0, 586))
    }

    print('Training and Testing on {} dataset...'.format(config.dataset))

    SavePath = config.svpath
    svPath = SavePath + config.dataset + '_' + str(config.vesion) + '_' + str(config.seed)
    os.makedirs(svPath, exist_ok=True)

    if config.seed == 0:
        pass
    else:
        print('we are using the seed = {}'.format(config.seed))
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    # 划分训练集/测试集
    total_num_images = img_num[config.dataset]
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

    solver = BaseVIT(config, device, folder_path[config.dataset], train_index, test_index)

    srcc, plcc = solver.test_ckpt(svPath, config.seed)
    print(f'SROCC={srcc},PLCC={plcc}')


if __name__ == '__main__':

    config = Configs()
    print(config)

    if torch.cuda.is_available():
        if len(config.gpunum) == 1:
            device = torch.device("cuda", index=int(config.gpunum))
        else:
            device = torch.device("cpu")

    main(config, device)
