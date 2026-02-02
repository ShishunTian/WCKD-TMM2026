import csv
import logging
import sys
import random
import numpy as np

from args import Configs
from data_loader import get_data
import torch
from tqdm import tqdm
from model import Net
from scipy import stats
import scipy
from scatter import plot_scatter

config = Configs()
device = torch.device("cuda", index=int(config.gpunum))
total_epoch = config.epochs
initial_lr = config.lr

seed = config.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dataset = config.dataset
dataset_dict = {'LIVE': 0, 'CSIQ': 1, 'bid': 2, 'CLIVE': 3, 'Koniq-10k': 4, 'Kadid-10k': 5, 'TID2013': 6}

train_loaders, val_loaders, test_loaders = get_data(dataset_dict[dataset])

sv_path = config.sv_path + dataset

model = Net(config).to(device)
model.load_state_dict(torch.load(sv_path + '/CovIQA_' + dataset + '.pth')['net'], strict=False)

model.eval()
pred_scores = []
gt_scores = []

pbartest = tqdm(test_loaders, leave=False, file=sys.stdout)
for sample_batched in pbartest:
    x, gmos = sample_batched['I'], sample_batched['mos']
    x = x.to(device)
    gmos = gmos.to(device)

    with torch.no_grad():
        batch_size = x.size(0)
        num_patch = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        pred_quality = model.forward(x)
        pred_quality = pred_quality.view(batch_size, num_patch, -1).mean(1)
        pred_scores = pred_scores + pred_quality.squeeze(-1).cpu().tolist()
        gt_scores = gt_scores + gmos.cpu().tolist()

srcc = scipy.stats.mstats.spearmanr(x=gt_scores, y=pred_scores)[0]
plcc = scipy.stats.mstats.pearsonr(x=gt_scores, y=pred_scores)[0]

# dataPath = sv_path + '/test_prediction_gt_{}_{}.csv'.format(dataset, str(seed))
# with open(dataPath, 'w') as f:
#     writer = csv.writer(f)
#     pred_scores_flat = [item[0] if isinstance(item, list) else item for item in pred_scores]
#     writer.writerows(zip(pred_scores, gt_scores))
#
# # logging the performance
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(sv_path + '/LogPerformance_{}_{}.log'.format(dataset, str(seed)))
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# Dataset = dataset
# logger.info(Dataset)
# PrintToLogg = 'Best PLCC: {}, SROCC: {}'.format(srcc, plcc)
# logger.info(PrintToLogg)
# logger.info('---------------------------')

print(f'SROCC: {srcc:.4f}, PLCC: {plcc:.4f},')

# plot_scatter(pred_scores, gt_scores, dataset)