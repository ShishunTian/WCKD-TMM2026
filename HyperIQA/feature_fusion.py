import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=2):
        super(CrossAttentionFusion, self).__init__()
        # 定义一个多头自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, quality_features, segmentation_features):
        quality_features = quality_features.permute(1, 0, 2)  # [B, HW, C] -> [HW, B, C]
        segmentation_features = segmentation_features.permute(1, 0, 2)

        # 将 quality_features 作为 Query，segmentation_features 作为 Key 和 Value
        attn_output, _ = self.attention(quality_features, segmentation_features, segmentation_features)
        attn_output = attn_output + quality_features

        return attn_output.permute(1, 0, 2)  # [HW, B, C] -> [B, HW, C]

if __name__ == '__main__':
    # 测试代码
    batch_size = 64
    seq_len = 197
    embed_dim = 768
    num_heads = 12

    # 假设我们有语义分割和质量特征
    seg_features = torch.randn(batch_size, seq_len, embed_dim)  # 形状: [batch_size, seq_len, embed_dim]
    qc_features = torch.randn(batch_size, seq_len, embed_dim)  # 形状: [batch_size, seq_len, embed_dim]

    # 初始化模型
    model = CrossAttentionFusion(embed_dim, num_heads)

    # 前向传播
    output = model(qc_features, seg_features)

    # 输出形状
    print(output.shape)  # 形状应该是 [batch_size, 2*seq_len, embed_dim]
