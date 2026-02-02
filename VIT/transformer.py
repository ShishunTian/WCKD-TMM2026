import csv
import json

import numpy as np
import torch.nn
import torchvision.models
from tqdm import tqdm
import sys
from scipy import stats

import data_loader
from instance_whitening import CovMatrix
from feature_fusion import CrossAttentionFusion

class Net(torch.nn.Module):
    def __init__(self, config, device):
        super(Net, self).__init__()

        self.device = device
        self.config = config

        self.net = torchvision.models.vit_b_16(pretrained=True).to(device)
        self.net.heads = torch.nn.Identity()

        self.heads = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 1)
        ).to(device)

        if self.config.cov or self.config.cross_correlation:

            self.layer_list = [0, 11]

            #加载dinov2
            self.dinov2 = torch.hub.load('./dinov2', 'dino_vitb16', source='local').to(self.device)
            for param in self.dinov2.parameters():
                param.requires_grad = False

            #注册钩子到每个编码器层
            self.features = []
            def hook_fn(module, input, output):
                self.features.append(output)
            for i in self.layer_list:
                layer_of_interest = self.net.encoder.layers[i]
                layer_of_interest.register_forward_hook(hook_fn)

            self.dino_features = []

        if self.config.cross_fusion:
            self.attn = CrossAttentionFusion()

        if self.config.fusion:
            self.fuse_linear = torch.nn.Linear(1536, 768).to(device)

    def forward(self, x):
        if self.config.cov or self.config.cross_correlation:
            self.features.clear()
            self.dino_features.clear()

            iqa_feature = self.net(x)
            dino_features = self.dinov2.get_intermediate_layers(x, self.layer_list)
            self.dino_features = dino_features

            if self.config.cross_fusion:
                attn_features = self.attn(self.features[-1], dino_features[-1])
                pred = self.heads(attn_features)
            elif self.config.fusion:
                dino_feat = dino_features[-1].mean(dim=1)
                fused_feat = torch.cat([iqa_feature, dino_feat], dim=1)
                fused_feat = self.fuse_linear(fused_feat)
                pred = self.heads(fused_feat)
            else:
                pred = self.heads(iqa_feature)
            return pred
        else:
            iqa_feature = self.net(x)
            pred = self.heads(iqa_feature)
            return pred

    def get_features(self):
        return self.features, self.dino_features

class BaseVIT(object):
    def __init__(self, config, device, datapath, train_idx, test_idx):
        super(BaseVIT, self).__init__()

        self.device = device
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.l1_loss = torch.nn.L1Loss()
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.droplr = config.droplr
        self.config = config

        self.net = Net(config,device).to(device)

        self.paras = [{'params': self.net.parameters(), 'lr': self.lr}]
        self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.solver, T_0=3, T_mult=2, eta_min=1e-7)

        train_loader = data_loader.DataLoader(config.dataset, datapath,
                                              train_idx, config.patch_size,
                                              config.train_patch_num,
                                              batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, datapath,
                                             test_idx, config.patch_size,
                                             config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        # cross_dataset = 'tid2013'
        # cross_path = '/data/hlz/IQA_Database/TID2013/'
        # with open('/data/hlz/VIT/Origin/' + cross_dataset + '_1_2024/test_index_1_2024.json','r') as file:
        #     cross_idx = json.load(file)
        # self.cross_test_loader = data_loader.DataLoader(cross_dataset, cross_path,
        #                                                 cross_idx, config.patch_size,
        #                                                 config.test_patch_num, istrain=False)
        # self.cross_test_data = self.cross_test_loader.get_data()

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
                    vit_features, dino_features = self.net.get_features()
                    for i in range(len(vit_features)):
                        cov_matrix_layer = CovMatrix(dim=768, clusters=3, device=self.device)

                        eps = 1e-5

                        IQA_feature = vit_features[i].permute(0,2,1)
                        dino_feature = dino_features[i].permute(0,2,1)

                        B,C,HW = IQA_feature.shape

                        eye, reverse_eye = cov_matrix_layer.get_eye_matrix()

                        dist_cor = torch.bmm(IQA_feature, IQA_feature.transpose(1, 2)).div(HW - 1) + (eps * eye)
                        scene_cor = torch.bmm(dino_feature, dino_feature.transpose(1, 2)).div(HW - 1) + (eps * eye)

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
                    vit_features, dino_features = self.net.get_features()
                    for i in range(len(vit_features)):

                        IQA_feature = vit_features[i].permute(0,2,1)
                        dino_feature = dino_features[i].permute(0,2,1)

                        B, C, HW = IQA_feature.shape

                        cross_corr = torch.bmm(IQA_feature, dino_feature.transpose(1, 2)).div(HW)

                        diagonal_elements = torch.diagonal(cross_corr,dim1=1,dim2=2)
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
                                                                                             sum(epoch_loss) / len(
                                                                                                 epoch_loss),
                                                                                             train_srcc, test_srcc,
                                                                                             test_plcc))
            if (epochnum + 1) == self.droplr \
                or (epochnum + 1) == (2 * self.droplr) \
                or (epochnum + 1) == (3 * self.droplr) \
                or (epochnum + 1) == (4 * self.droplr) \
                or (epochnum + 1) == (5 * self.droplr) \
                or (epochnum + 1) == (6 * self.droplr):
                self.lr = self.lr / 5
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
        new_state_dict = {k.replace("net.heads.", "heads.") if k.startswith("net.heads.") else k: v
                          for k, v in checkpoint.items()}
        self.net.load_state_dict(new_state_dict, strict=False)

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

    # def cross_test(self, svPath, seed):
    #     self.net.load_state_dict(torch.load(svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed))))
    #     self.net.eval()
    #     pred_scores = []
    #     gt_scores = []
    #     pbartest = tqdm(self.cross_test_data, leave=False)
    #     with torch.no_grad():
    #         for img, label in pbartest:
    #             img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
    #             label = torch.as_tensor(label.to(self.device)).requires_grad_(False)
    #
    #             pred = self.net(img)
    #
    #             pred_scores = pred_scores + pred.cpu().tolist()
    #             gt_scores = gt_scores + label.cpu().tolist()
    #
    #     pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
    #     gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
    #     test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    #     test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    #     return test_srcc, test_plcc



if __name__ == '__main__':
    import torch
    from torchvision.models import vit_b_16
    from args import Configs
    from ptflops import get_model_complexity_info
    torch.manual_seed(2024)

    config = Configs()

    if torch.cuda.is_available():
        if len(config.gpunum) == 1:
            device = torch.device("cuda", index=int(config.gpunum))
        else:
            device = torch.device("cpu")

    # net = Net(config, device).to(device)
    # x = torch.randn(1, 3, 224, 224).to(device)

    # total_params = sum(p.numel() for p in net.parameters())
    # trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # total_m = total_params / 1e6
    # trainable_m = trainable_params / 1e6
    # print(f"Total Parameters: {total_m:.2f}M")
    # print(f"Trainable Parameters: {trainable_m:.2f}M")

    # with torch.no_grad():
    #     torch.cuda.synchronize()
    #     for _ in range(10):
    #         _,_,_ = net(x)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     for _ in range(100):
    #         _,_,_= net(x)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #
    #     avg_infer_time = (end - start) / 100 * 1000
    #     print(f"Inference time: {avg_infer_time:.2f} ms")

    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
    #     print(f'FLOPs: {macs}, Params: {params}')


