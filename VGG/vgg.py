import csv
import json
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
from tqdm import tqdm
import sys
from scipy import stats
import data_loader
from DeepLabV1.vgg import VGG16_LargeFOV, extract_res
import time

from instance_whitening import CovMatrix
from feature_fusion import CrossAttentionFusion

# class Net(torch.nn.Module):
#     def __init__(self, config, device):
#         super(Net, self).__init__()
#
#         self.device = device
#         self.config = config
#
#         self.net = torchvision.models.vgg16(pretrained=True).to(device)
#         self.net.classifier = torch.nn.Identity()
#
#         self.reduce = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5)
#         ).to(device)
#
#         self.heads = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, 1)
#         ).to(device)
#
#         if self.config.cross_correlation or self.config.cov:
#             # 加载deeplabv1
#             deeplabv1 = VGG16_LargeFOV().to(device)
#             deeplabv1.load_state_dict(torch.load('./DeepLabV1/deeplabv1.pth', map_location=torch.device('cpu')), strict=True)
#
#             for param in deeplabv1.parameters():
#                 param.requires_grad = False
#             self.deeplabv1 = deeplabv1
#
#             #注册钩子到每个层
#             self.features = []
#             def hook_fn(module, input, output):
#                 self.features.append(output)
#
#             for name, module in self.net.features.named_children():
#                 if name in ["3", "8", "15", "22", "29"]:
#                     module.register_forward_hook(hook_fn)
#
#         if self.config.fusion:
#             self.avg_pool = nn.AdaptiveAvgPool2d((7, 7)).to(device)
#             self.fuse_linear = nn.Linear(4096 * 2, 4096).to(device)
#
#         if self.config.cross_fusion:
#             self.avg_pool = nn.AdaptiveAvgPool2d((7, 7)).to(device)
#             self.attn = CrossAttentionFusion()
#
#
#     def forward(self, x):
#         if self.config.cross_correlation or self.config.cov:
#             self.features.clear()
#             iqa_feature = self.net(x)
#             seg_features = extract_res(x, self.deeplabv1)
#
#             if self.config.fusion:
#                 seg_feat = self.avg_pool(seg_features[4])
#                 seg_feat = torch.flatten(seg_feat, 1)
#
#                 seg_feat = self.reduce(seg_feat)
#                 iqa_feature = self.reduce(iqa_feature)
#
#                 fused_feat = torch.cat([iqa_feature, seg_feat], dim=1)
#                 fused_feat = self.fuse_linear(fused_feat)
#                 pred = self.heads(fused_feat)
#             elif self.config.cross_fusion:
#                 iqa_feat = self.features[4]
#                 seg_feat = seg_features[4]
#                 b, c, h, w = iqa_feat.size()
#                 iqa_feat_seq = iqa_feat.view(b, c, -1).permute(0, 2, 1)  # [B, C, H, W] -> [B, HW, C]
#                 seg_feat_seq = seg_feat.view(b, c, -1).permute(0, 2, 1)
#
#                 attn_features = self.attn(iqa_feat_seq, seg_feat_seq)
#                 attn_features = attn_features.permute(0, 2, 1).view(b, c, h, w)  # [B, HW, C] -> [B, C, H, W]
#                 attn_features = self.avg_pool(attn_features)
#                 attn_features = torch.flatten(attn_features, 1)
#                 attn_features = self.reduce(attn_features)
#                 pred = self.heads(attn_features)
#             else:
#                 iqa_feature = self.reduce(iqa_feature)
#                 pred = self.heads(iqa_feature)
#             return pred, self.features, seg_features
#         else:
#             iqa_feature = self.net(x)
#             iqa_feature = self.reduce(iqa_feature)
#             pred = self.heads(iqa_feature)
#             return pred

class Net(torch.nn.Module):
    def __init__(self, config, device):
        super(Net, self).__init__()

        self.device = device
        self.config = config

        self.net = torchvision.models.vgg16(pretrained=True).to(device)
        self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.net.classifier = torch.nn.Identity()

        self.heads = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 1)
        ).to(device)

        # 存储需要提取的IQA模型特征
        self.features = []

        if self.config.cov or self.config.cross_correlation or self.config.fusion or self.config.cross_fusion:
            # 加载deeplabv1
            deeplabv1 = VGG16_LargeFOV().to(device)
            deeplabv1.load_state_dict(torch.load('./DeepLabV1/deeplabv1.pth', map_location=torch.device('cpu')), strict=True)

            for param in deeplabv1.parameters():
                param.requires_grad = False
            self.deeplabv1 = deeplabv1
            self.seg_features = []
            def hook_fn(module, input, output):
                self.features.append(output)

            for name, module in self.net.features.named_children():
                if name in ["3", "8", "15", "22", "29"]:
                    module.register_forward_hook(hook_fn)

        if self.config.fusion:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            self.fuse_linear = nn.Linear(512 * 2, 512).to(device)

        if self.config.cross_fusion:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            self.attn = CrossAttentionFusion()


    def forward(self, x):
        self.features.clear()
        iqa_feature = self.net(x)

        if self.config.cov or self.config.cross_correlation or self.config.fusion or self.config.cross_fusion:
            seg_features = extract_res(x, self.deeplabv1)
            self.seg_features = seg_features

        if self.config.fusion:
            seg_feat = self.avg_pool(self.seg_features[4])
            seg_feat = torch.flatten(seg_feat, 1)

            fused_feat = torch.cat([iqa_feature, seg_feat], dim=1)
            fused_feat = self.fuse_linear(fused_feat)
            pred = self.heads(fused_feat)
        elif self.config.cross_fusion:
            iqa_feat = self.features[4]
            seg_feat = self.seg_features[4]
            b, c, h, w = iqa_feat.size()
            iqa_feat_seq = iqa_feat.view(b, c, -1).permute(0, 2, 1)  # [B, C, H, W] -> [B, HW, C]
            seg_feat_seq = seg_feat.view(b, c, -1).permute(0, 2, 1)

            attn_features = self.attn(iqa_feat_seq, seg_feat_seq)
            attn_features = attn_features.permute(0, 2, 1).view(b, c, h, w)  # [B, HW, C] -> [B, C, H, W]
            attn_features = self.avg_pool(attn_features)
            attn_features = torch.flatten(attn_features, 1)
            pred = self.heads(attn_features)
        else:
            pred = self.heads(iqa_feature)
        return pred

    def get_features(self):
        return self.features, self.seg_features


class Base_VGG(object):
    def __init__(self, config, device, datapath, train_idx, test_idx):
        super(Base_VGG, self).__init__()

        self.device = device
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.l1_loss = torch.nn.L1Loss()
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.droplr = config.droplr
        self.config = config

        self.net = Net(config, device).to(device)

        self.paras = [{'params': self.net.parameters(), 'lr': self.lr}]
        self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, datapath,
                                              train_idx, config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, datapath,
                                             test_idx, config.patch_size,
                                             config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self, seed, svPath, pretrained = False):
        best_srcc = 0.0
        best_plcc = 0.0
        if pretrained:
            self.net.load_state_dict(torch.load(svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed)), map_location=torch.device('cpu')))
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        results = {}

        performPath = svPath + '/' + 'PLCC_SRCC_' + str(self.config.vesion) + '_' + str(seed) + '.json'
        with open(performPath, 'w') as json_file2:
            json.dump({}, json_file2)

        for epochnum in range(self.epochs):
            self.net.train()
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            pbar = tqdm(self.train_data, leave=False, file=sys.stdout)

            for img, label in pbar:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                self.net.zero_grad()

                pred = self.net(img)

                pred_scores = pred_scores + pred.flatten().cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())
                loss = loss_qa

                if self.config.cov:
                    loss_ccs = 0.0
                    cov_delta = 0.5
                    iqa_features, seg_features = self.net.get_features()
                    for idx, dim in enumerate([64, 128, 256, 512, 512], 0):
                        cov_matrix_layer = CovMatrix(dim=dim, clusters=3, device=self.device)
                        eps = 1e-5

                        iqa_feature = iqa_features[idx]
                        seg_feature = seg_features[idx]

                        B,C,H,W = iqa_feature.shape
                        HW = H * W

                        iqa_feature = iqa_feature.view(B,C,-1)
                        seg_feature = seg_feature.view(B,C,-1)

                        eye, reverse_eye = cov_matrix_layer.get_eye_matrix()

                        dist_cor = torch.bmm(iqa_feature, iqa_feature.transpose(1, 2)).div(HW - 1) + (eps * eye)
                        scene_cor = torch.bmm(seg_feature, seg_feature.transpose(1, 2)).div(HW - 1) + (eps * eye)

                        scene_elements = scene_cor * reverse_eye
                        dist_elements = dist_cor * reverse_eye

                        diff_mat = torch.abs(scene_elements - dist_elements)
                        for sample in range(diff_mat.size(0)):
                            cov_matrix_layer.set_variance_of_covariance(diff_mat[sample])

                        cov_matrix_layer.get_mask_matrix()

                        loss_cc = cov_matrix_layer.instance_whitening_loss(dist_cor, diff_mat)
                        loss_ccs += loss_cc
                    loss += loss_ccs * cov_delta

                if self.config.cross_correlation:
                    loss_decorrelation = 0.0
                    decorr_delta = 0.5
                    iqa_features, seg_features = self.net.get_features()
                    for idx, dim in enumerate([64, 128, 256, 512, 512], 0):
                        iqa_feature = iqa_features[idx]
                        seg_feature = seg_features[idx]
                        B, C, H, W = iqa_feature.shape
                        HW = H * W

                        iqa_feature = iqa_feature.view(B, C, -1) # (B, C, HW)
                        seg_feature = seg_feature.view(B, C, -1)

                        iqa_feature = torch.nn.functional.normalize(iqa_feature, p=2, dim=1)
                        seg_feature = torch.nn.functional.normalize(seg_feature, p=2, dim=1)

                        cross_corr = torch.bmm(iqa_feature, seg_feature.transpose(1, 2)).div(HW)

                        diagonal_elements = torch.diagonal(cross_corr, dim1=1, dim2=2)
                        loss_decorr = torch.mean(torch.abs(diagonal_elements))
                        loss_decorrelation += loss_decorr
                    loss += loss_decorrelation * decorr_delta

                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            modelPath = svPath + '/model_{}_{}_{}'.format(str(self.config.vesion), str(seed), epochnum)
            torch.save(self.net.state_dict(), modelPath)

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            test_srcc, test_plcc = self.test(self.test_data, epochnum, svPath, seed)

            results[epochnum] = (test_srcc, test_plcc)

            with open(performPath, "r+") as file:
                data = json.load(file)
                data.update(results)
                file.seek(0)
                json.dump(data, file)

            if test_srcc > best_srcc:
                modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed))
                torch.save(self.net.state_dict(), modelPathbest)
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('{}\t\t{:4.3f}\t\t{:4.4f}\t\t{:4.4f}\t\t{:4.4f}'.format(epochnum + 1,
                                                                          sum(epoch_loss) / len( epoch_loss),
                                                                          train_srcc, test_srcc, test_plcc))

            if (epochnum + 1) == self.droplr or (epochnum + 1) == (2 * self.droplr) or (epochnum + 1) == (
                    3 * self.droplr):
                self.lr = self.lr / 10
                self.paras = [{'params': self.net.parameters(), 'lr': self.lr}]
                self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))
        return best_srcc, best_plcc

    def test(self, data, epochnum, svPath, seed, pretrained=0):
        if pretrained:
            self.net.load_state_dict(torch.load(svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed))))
        self.net.eval()
        pred_scores = []
        gt_scores = []

        pbartest = tqdm(data, leave=False)

        with torch.no_grad():
            for img, label in pbartest:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                pred = self.net(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        # if not pretrained:
        dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion), str(seed), epochnum)
        with open(dataPath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(pred_scores, gt_scores))

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        return test_srcc, test_plcc

    def test_ckpt(self, svPath, seed):
        checkpoint = torch.load(svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed)), map_location=torch.device('cuda:' + self.config.gpunum))
        self.net.load_state_dict(checkpoint, strict=False)
        self.net.eval()
        pred_scores = []
        gt_scores = []
        pbartest = tqdm(self.test_data, leave=False)
        with torch.no_grad():
            for img, label in pbartest:
                img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

                pred = self.net(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        return test_srcc, test_plcc

if __name__ == '__main__':
    import torch
    from args import Configs
    import torchvision
    from ptflops import get_model_complexity_info

    torch.manual_seed(2024)

    config = Configs()

    device = torch.device("cuda", index=int(config.gpunum))

    net = Net(config, device).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6
    print(f"Total Parameters: {total_m:.2f}M")
    print(f"Trainable Parameters: {trainable_m:.2f}M")

    x = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        torch.cuda.synchronize()
        for _ in range(10):
            _,_,_ = net(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _,_,_= net(x)
        torch.cuda.synchronize()
        end = time.time()

        avg_infer_time = (end - start) / 100 * 1000
        print(f"Inference time: {avg_infer_time:.2f} ms")

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
        print(f'FLOPs: {macs}, Params: {params}')

    # net = torchvision.models.vgg16(pretrained=True).features.to(device)
    # x = torch.randn(1, 3, 224, 224).to(device)
    # result = net(x)
    # print(result)