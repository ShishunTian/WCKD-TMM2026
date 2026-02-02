import torch as torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np
import os
from resnet_modify import resnet50
from DeepLabV3.network import modeling

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config
        device = torch.device("cuda", index=int(config.gpunum))

        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)

        dim_modelt = 3840

        modelpretrain = models.resnet50(pretrained=True)
        torch.save(modelpretrain.state_dict(), 'modelpretrain')
        self.model = resnet50()
        self.model.load_state_dict(torch.load('modelpretrain'), strict=True)
        os.remove("modelpretrain")

        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))
        self.drop2d = nn.Dropout(p=0.1)

        self.fc = nn.Linear(dim_modelt, self.model.fc.in_features)
        self.fc_quality = nn.Linear(self.model.fc.in_features, 1)

        deeplabv3 = modeling.deeplabv3_resnet50()
        ckpt = torch.load('/data/hlz/pretrained_model/deeplabv3_resnet50_coco-cd0a2569.pth', map_location=torch.device('cpu'))
        deeplabv3.load_state_dict(ckpt, strict=False)
        deeplabv3 = deeplabv3.backbone.to(device)
        for param in deeplabv3.parameters():
            param.requires_grad = False
        self.deeplabv3 = deeplabv3

        self.deeplab_features = None
        self.IQA_feature = None

    def forward(self, x):
        out, layer1, layer2, layer3, layer4 = self.model(x)

        self.IQA_feature = {'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'layer4': layer4}
        self.deeplab_features = self.deeplabv3(x)

        if self.config.fusion:
            self.get_concatenated_features()

        layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
        layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
        layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2))))
        layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
        layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)
        layers = torch.flatten(self.avg7(layers), start_dim=1)
        layers = self.fc(layers)
        pred_quality = self.fc_quality(layers)
        return pred_quality

    def get_features(self):
        return self.IQA_feature, self.deeplab_features

    def get_concatenated_features(self):
        if self.IQA_feature is None or self.deeplab_features is None:
            return None
        
        concatenated_features = {}
        for layer_name in self.IQA_feature.keys():
            if layer_name in self.deeplab_features:
                concatenated_features[layer_name] = torch.cat(
                    (self.IQA_feature[layer_name], self.deeplab_features[layer_name]), 
                    dim=1
                )
        
        return concatenated_features
