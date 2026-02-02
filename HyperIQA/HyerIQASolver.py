import json
import torch
from scipy import stats
import numpy as np
import models
import data_loader
from tqdm import tqdm
import sys
import csv

from instance_whitening import CovMatrix

class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx, device):

        self.device = device
        self.config = config

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, device, config).to(device)
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().to(device)

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self, svPath, seed):
        best_srcc = 0.0
        best_plcc = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')

        results = {}
        performPath = svPath + '/' +'PLCC_SRCC_' + str(self.config.vesion) + '_' + str(seed) + '.json'
        with open(performPath, 'w') as json_file2:
            json.dump({}, json_file2)

        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            pbar = tqdm(self.train_data, leave=False, file=sys.stdout)

            for img, label in pbar:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                self.solver.zero_grad()

                paras = self.model_hyper(img)

                model_target = models.TargetNet(paras).to(self.device)
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
                loss = loss_qa

                if self.config.cov:
                    loss_ccs = 0.0
                    cov_delta = 0.5
                    iqa_features, seg_features = self.model_hyper.get_features()

                    for idx, dim in enumerate([256, 512, 1024, 2048], 0):
                        cov_matrix_layer = CovMatrix(dim=dim, clusters=3, device=self.device)
                        eps = 1e-5

                        IQA_feature = iqa_features['layer' + str(idx + 1)]
                        deeplab_feature = seg_features['layer' + str(idx + 1)]

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
                        loss_ccs += loss_cc
                    loss += loss_ccs * cov_delta

                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            modelPath = svPath + '/model_{}_{}_{}'.format(str(self.config.vesion), str(seed), t)
            torch.save(self.model_hyper.state_dict(), modelPath)

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data, svPath, seed, t)

            results[t] = (test_srcc, test_plcc)
            with open(performPath, "r+") as file:
                data = json.load(file)
                data.update(results)
                file.seek(0)
                json.dump(data, file)

            if test_srcc > best_srcc:
                modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed))
                torch.save(self.model_hyper.state_dict(), modelPathbest)
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data, svPath, seed, epochnum):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        pbartest = tqdm(data, leave=False, file=sys.stdout)

        with torch.no_grad():
            for img, label in pbartest:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                paras = self.model_hyper(img)

                model_target = models.TargetNet(paras).to(self.device)
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

                pred_scores = pred_scores + pred.flatten().cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion), str(seed), epochnum)
        with open(dataPath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(pred_scores, gt_scores))

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)
        return test_srcc, test_plcc
    def test_ckpt(self, svPath, seed):
        checkpoint = torch.load(svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed)), map_location=torch.device('cuda:' + self.config.gpunum))
        checkpoint = self.load_compatible_checkpoint(checkpoint)
        self.model_hyper.load_state_dict(checkpoint, strict=True)

        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        pbartest = tqdm(self.test_data, leave=False, file=sys.stdout)

        with torch.no_grad():
            for img, label in pbartest:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                paras = self.model_hyper(img)

                model_target = models.TargetNet(paras).to(self.device)
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

                pred_scores = pred_scores + pred.flatten().cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        return test_srcc, test_plcc
    def load_compatible_checkpoint(self, checkpoint):
        """
        处理新旧模型结构兼容性问题
        如果checkpoint包含conv0，则将其与conv1合并
        """
        model_state = self.model_hyper.state_dict()
        updated_checkpoint = checkpoint.copy()

        # 检查是否为旧版模型结构（包含conv0）
        has_conv0 = any(key.startswith('conv0.') for key in checkpoint.keys())

        if has_conv0:
            # 从conv0获取第一层参数
            if 'conv0.0.weight' in checkpoint and 'conv0.0.bias' in checkpoint:
                # 更新conv1.0的参数（对应旧版conv0.0）
                updated_checkpoint['conv1.0.weight'] = checkpoint['conv0.0.weight']
                updated_checkpoint['conv1.0.bias'] = checkpoint['conv0.0.bias']
            # 从conv1获取后续层参数（旧版conv1结构）
            if 'conv1.0.weight' in checkpoint and 'conv1.0.bias' in checkpoint:
                # 更新conv1.2的参数（对应旧版conv1.0）
                updated_checkpoint['conv1.2.weight'] = checkpoint['conv1.0.weight']
                updated_checkpoint['conv1.2.bias'] = checkpoint['conv1.0.bias']
            if 'conv1.2.weight' in checkpoint and 'conv1.2.bias' in checkpoint:
                # 更新conv1.4的参数（对应旧版conv1.2）
                updated_checkpoint['conv1.4.weight'] = checkpoint['conv1.2.weight']
                updated_checkpoint['conv1.4.bias'] = checkpoint['conv1.2.bias']
            if 'conv0.0.weight' in updated_checkpoint:
                del updated_checkpoint['conv0.0.weight']
            if 'conv0.0.bias' in updated_checkpoint:
                del updated_checkpoint['conv0.0.bias']

        return updated_checkpoint
