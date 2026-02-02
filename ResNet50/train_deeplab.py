import os
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
from instance_whitening import CovMatrix

from weight_methods import WeightMethods


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

model = Net(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_srcc = 0.0
best_plcc = 0.0

l1_loss = torch.nn.L1Loss()

weighting_method = WeightMethods(method='dwa', n_tasks=2, alpha=1.5, temp=2.0, n_train_batch=16, n_epochs=total_epoch, main_task=0, device=device)

sv_path = os.path.join(config.sv_path, dataset)
os.makedirs(sv_path, exist_ok=True)
sv_path = os.path.join(sv_path, 'CovIQA_'+dataset+'.pth')

def save_model(srcc, plcc, best_srcc, best_plcc, sv_path):
    if srcc > best_srcc:
        print('=> Save checkpoint')
        best_srcc = srcc
        best_plcc = plcc
        model.eval()
        ckpt = {
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(ckpt, sv_path)
        model.train()
    print(f'BEST SROCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}')
    return best_srcc, best_plcc


print(f'+====================+ Training On {dataset} +====================+')
for epoch in range(0, total_epoch):
    epoch_loss = []
    pred_scores = []
    gt_scores = []
    model.train()
    print(f'+====================+ Training Epoch: {epoch} +====================+')
    pbar = tqdm(train_loaders, leave=False, file=sys.stdout)
    for sample_batched in pbar:
        x, gmos = sample_batched['I'], sample_batched['mos']

        x = x.to(device).requires_grad_(False)
        gmos = gmos.to(device).requires_grad_(False)

        optimizer.zero_grad()

        batch_size = x.size(0)
        num_patch = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        pred_quality = model.forward(x)

        pred_quality = pred_quality.view(batch_size, num_patch, -1).mean(1)
        pred_scores = pred_scores + pred_quality.flatten().cpu().tolist()
        gt_scores = gt_scores + gmos.cpu().tolist()

        loss_quality = l1_loss(pred_quality.squeeze(), gmos.float().detach())

        if config.cov:
            IQA_features, deeplab_features = model.get_features()
            loss_cov = 0.0
            cov_delta = 0.5
            for idx, dim in enumerate([256, 512, 1024, 2048], 0):
                cov_matrix_layer = CovMatrix(dim=dim, clusters=3, device=device)
                eps = 1e-5

                IQA_feature = IQA_features['layer' + str(idx+1)]
                deeplab_feature = deeplab_features['layer' + str(idx+1)]

                B, C, H, W = IQA_feature.shape
                HW = H * W
                IQA_feature = IQA_feature.view(B, C, -1)
                deeplab_feature = deeplab_feature.view(B, C, -1)

                eye, reverse_eye = cov_matrix_layer.get_eye_matrix()

                scene_cor = torch.bmm(deeplab_feature, deeplab_feature.transpose(1, 2)).div(HW - 1) + (eps * eye)
                dist_cor = torch.bmm(IQA_feature, IQA_feature.transpose(1, 2)).div(HW - 1) + (eps * eye)

                scene_elements = scene_cor * reverse_eye
                dist_elements = dist_cor * reverse_eye

                diff_mat = torch.abs(scene_elements - dist_elements)
                for sample in range(diff_mat.size(0)):
                    cov_matrix_layer.set_variance_of_covariance(diff_mat[sample])

                cov_matrix_layer.get_mask_matrix()

                loss_cc = cov_matrix_layer.instance_whitening_loss(dist_cor, diff_mat)
                loss_cov += loss_cc
            loss = [loss_quality, cov_delta * loss_cov]
            loss = weighting_method.backwards(loss, epoch=epoch, logsigmas=None, shared_parameters=None, last_shared_params=None, returns=True)
        else:
            loss = loss_quality
            loss.backward()

        optimizer.step()
        scheduler.step()
        epoch_loss.append(loss.item())

    train_srcc = scipy.stats.mstats.spearmanr(x=gt_scores, y=pred_scores)[0]
    print(f'Loss: {sum(epoch_loss) / len(epoch_loss)}')
    print(f'Train_SRCC: {train_srcc}')
    print(weighting_method.method.lambda_weight[:, epoch])

    print(f'+====================+ Testing Epoch: {epoch} +====================+')
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
        pred_scores = pred_scores + pred_quality.cpu().tolist()
        gt_scores = gt_scores + gmos.cpu().tolist()

    srcc = scipy.stats.mstats.spearmanr(x=gt_scores, y=pred_scores)[0]
    plcc = scipy.stats.mstats.pearsonr(x=gt_scores, y=pred_scores)[0]
    print(f'SROCC: {srcc:.4f}, PLCC: {plcc:.4f}')
    best_srcc, best_plcc = save_model(srcc, plcc, best_srcc, best_plcc, sv_path)