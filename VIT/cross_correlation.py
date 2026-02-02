import torch
import torch.nn.functional as F

def compute_cross_correlation(iqa_features, seg_features):
    batch_size, channels, height, width = iqa_features.shape

    iqa_features_flat = iqa_features.view(batch_size, channels, -1)
    seg_features_flat = seg_features.view(batch_size, channels, -1)

    iqa_features_flat = F.normalize(iqa_features_flat, dim=2)
    seg_features_flat = F.normalize(seg_features_flat, dim=2)

    # 计算通道间的互相关矩阵 (batch_size, channels, channels)
    cross_corr = torch.bmm(iqa_features_flat, seg_features_flat.transpose(1, 2)) / (height * width)
    return cross_corr


def decorrelate_features(cross_corr):
    # 获取对角线元素 (每个通道对自身的相关性)
    batch_size, channels, _ = cross_corr.shape
    diagonal_elements = torch.diagonal(cross_corr, dim1=1, dim2=2)  # (batch_size, channels)

    # 计算L1损失，仅最小化对角线元素
    loss = torch.mean(torch.abs(diagonal_elements))
    return loss


# 示例数据
iqa_features = torch.randn(16, 768, 14, 14)
seg_features = torch.randn(16, 768, 14, 14)

# 计算互相关特征和解耦损失
cross_corr = compute_cross_correlation(iqa_features, seg_features)
decorrelation_loss = decorrelate_features(cross_corr)

print("L1 Decorrelation Loss:", decorrelation_loss.item())