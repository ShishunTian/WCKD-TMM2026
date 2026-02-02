import logging
import os
import json
import random
import numpy as np
from HyerIQASolver import HyperIQASolver
import torch
import warnings
from args import Configs
warnings.filterwarnings("ignore")

def main(config, device):
    folder_path = {
        'live': config.datapath,
        'csiq': config.datapath,
        'tid2013': config.datapath,
        'kadid10k': config.datapath,
        'livec': config.datapath,
        'koniq-10k': config.datapath,
        'bid': config.datapath,
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'kadid10k': list(range(0, 80))
    }

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))

    svPath = config.svpath + config.dataset + '_' + str(config.vesion) + '_' + str(config.seed)
    os.makedirs(svPath, exist_ok=True)

    print('we are using the seed = {}'.format(config.seed))
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # 划分训练集/测试集
    sel_num = img_num[config.dataset]
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

    imgsTrainPath = svPath + '/' + 'train_index_' + str(config.vesion) + '_' + str(config.seed) + '.json'
    imgsTestPath = svPath + '/' + 'test_index_' + str(config.vesion) + '_' + str(config.seed) + '.json'
    with open(imgsTrainPath, 'w') as json_file2:
        json.dump(train_index, json_file2)
    with open(imgsTestPath, 'w') as json_file2:
        json.dump(test_index, json_file2)

    solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index, device)
    srcc, plcc = solver.train(svPath, config.seed)

    # logging the performance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(svPath + '/LogPerformance.log')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    Dataset = config.dataset
    logger.info(Dataset)

    PrintToLogg = 'Median SROCC: {}, PLCC: {}'.format(srcc, plcc)
    logger.info(PrintToLogg)
    logger.info('---------------------------')


if __name__ == '__main__':
    config = Configs()
    print(config)

    if torch.cuda.is_available():
            if len(config.gpunum)==1:
                device = torch.device("cuda", index=int(config.gpunum))
            else:
                device = torch.device("cpu")

    main(config, device)
