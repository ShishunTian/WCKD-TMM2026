import torch
import torch.nn as nn
from kmeans_pytorch import kmeans


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x

        return x, w


class CovMatrix:
    def __init__(self, dim, clusters=3, device=None):
        super(CovMatrix, self).__init__()

        self.device = device

        self.dim = dim
        self.i = torch.eye(dim, dim).to(self.device)
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).to(self.device)

        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.var_matrix = None
        self.mask_matrix = None
        self.clusters = clusters

        self.num_big = 0
        self.high_diff_mask_matrix = None

        self.loss = 0.0
        self.loss_big = 0.0

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def get_mask_matrix(self):
        if self.mask_matrix is None:
            self.set_mask_matrix()

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self):
        var_flatten = torch.flatten(self.var_matrix)
        var_non_zero = var_flatten[var_flatten != 0].unsqueeze(1)
        index = var_flatten.nonzero()

        clusters, cluster_centers = kmeans(X=var_non_zero, num_clusters=self.clusters, distance='euclidean', device=self.device)
        indices = torch.nonzero(clusters == torch.argmin(cluster_centers)).squeeze(1)
        index = index[indices].squeeze(1)

        # mask_matrix = torch.flatten(torch.ones(self.dim, self.dim).triu(diagonal=1)).to(self.device)
        # mask_matrix[index] = 0

        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).to(self.device))
        mask_matrix[index] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)

        #高差异项
        index = var_flatten.nonzero()
        high_indices = torch.nonzero(clusters == torch.argmax(cluster_centers)).squeeze(1)
        high_index = index[high_indices].squeeze(1)

        high_diff_mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).to(self.device))
        high_diff_mask_matrix[high_index] = 1
        self.high_diff_mask_matrix = high_diff_mask_matrix.view(self.dim, self.dim)
        self.num_big = torch.sum(self.high_diff_mask_matrix)

        self.var_matrix = None

    def set_variance_of_covariance(self, var_cov):
        if self.var_matrix is None:
            self.var_matrix = var_cov
        else:
            self.var_matrix = self.var_matrix + var_cov

    def instance_whitening_loss(self, dist_cor, diff_mat):
        for sample in range(dist_cor.size(0)):
            dist_cor[sample] = dist_cor[sample] * self.mask_matrix
        off_diag_sum = torch.sum(torch.abs(dist_cor), dim=(1, 2), keepdim=True)  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, self.num_sensitive), min=0)  # B X 1 X 1
        self.loss = torch.sum(loss)

        diff_mat = torch.mean(diff_mat, dim=0)
        min_val = diff_mat.min()
        max_val = diff_mat.max()
        diff_mat = (diff_mat - min_val) / (max_val - min_val + 1e-8)

        diff_masked = diff_mat * self.high_diff_mask_matrix
        self.loss_big = torch.nn.functional.l1_loss(diff_masked, self.high_diff_mask_matrix)
        return self.loss

    # def set_mask_matrix(self):
    # 	var_flatten = torch.flatten(self.var_matrix)
    #
    # 	cluster_ids_x, cluster_centers = kmeans(X=var_flatten.unsqueeze(1), num_clusters=3, device=self.device)
    # 	unique_values, counts = torch.unique(cluster_ids_x, return_counts=True)
    # 	most_frequent_value = unique_values[counts.argmax()].item()
    # 	num_sensitive = var_flatten.size()[0] - cluster_ids_x.tolist().count(most_frequent_value)  # 1: Insensitive Cov, 2~50: Sensitive Cov
    # 	_, indices = torch.topk(var_flatten, k=int(num_sensitive))
    #
    # 	mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim)).to(self.device)
    # 	mask_matrix[indices] = 1
    #
    # 	if self.mask_matrix is not None:
    # 		self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
    # 	else:
    # 		self.mask_matrix = mask_matrix.view(self.dim, self.dim)
    # 	self.num_sensitive = torch.sum(self.mask_matrix)
    #
    # 	self.var_matrix = None
    # 	self.count_var_cov = 0


# def instance_whitening_loss(f_cor, mask_matrix, num_remove_cov, diff_mat, high_mask_matrix, big):
#     B = f_cor.size(0)
#
#     for sample in range(B):
#         f_cor[sample] = f_cor[sample] * mask_matrix
#     off_diag_sum = torch.sum(torch.abs(f_cor), dim=(1, 2), keepdim=True)  # B X 1 X 1
#     loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
#     loss = torch.sum(loss)
#
#     diff_mat = torch.mean(diff_mat, dim=0)
#     min_val = diff_mat.min()
#     max_val = diff_mat.max()
#     diff_mat = (diff_mat - min_val) / (max_val - min_val + 1e-8)
#
#     diff_masked = diff_mat * high_mask_matrix
#     loss_big = torch.nn.functional.l1_loss(diff_masked, high_mask_matrix)
#
#     return loss

# def instance_whitening_loss(f_cor, mask_matrix, num_remove_cov):
#     B = f_cor.size(0)
#     for sample in range(B):
#         f_cor[sample] = f_cor[sample] * mask_matrix
#     off_diag_sum = torch.sum(torch.abs(f_cor), dim=(1,2), keepdim=True) # B X 1 X 1
#     loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
#     loss = torch.sum(loss)
#     return loss
