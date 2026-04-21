import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConceptGraph(nn.Module):
    def __init__(self, num_concepts=6, pool_size=7, sigma=0.8, out_features=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.pool_size = pool_size
        self.num_spatial_nodes = pool_size * pool_size  # 49
        self.total_nodes = num_concepts * self.num_spatial_nodes  # 6 * 49 = 294
        self.norm_layer = nn.BatchNorm2d(6)

        # 1. 静态空间距离衰减矩阵 D (不参与学习)
        Dxx = self._build_spatial_distance(pool_size, sigma)
        self.register_buffer("Dxx", Dxx)

        # 2. 静态医学先验矩阵 P_static (不参与学习)
        P_staticxx = self._build_prior_matrix()
        self.register_buffer("P_staticxx", P_staticxx)

        # 3. 半静态学习的核心：可学习的微调矩阵 P_delta
        # 初始化为全0，意味着训练初期完全依靠专家的 P_static
        self.P_delta = nn.Parameter(torch.zeros(num_concepts, num_concepts))

        # 微调幅度锁 (控制模型最多偏离专家经验多少，0.2是一个适中的值)
        self.alpha = 0.2

        # 融合更新率 (控制当前节点吸收多少外来信息)
        self.beta = 0.5

    def _build_spatial_distance(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([y.flatten(), x.flatten()], dim=1)
        dist_sq = torch.cdist(grid, grid, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma**2))

    def _build_prior_matrix(self):
        """医学先验连通性矩阵"""
        # 定义病灶概念和其先验权重
        # 0:HE, 1:EX, 2:MA, 3:SE, 4:VHE, 5:VOP
        # 其中 HE, EX, SE 是负相关（抑制）的通道
        P = torch.zeros((6, 6))

        # 自身传递
        P[0, 0] = 1
        P[1, 1] = 1
        P[2, 2] = 0.5
        P[3, 3] = 1
        P[4, 4] = 1
        P[5, 5] = 1
        # 互斥逻辑
        P[1, 2] = 0.6
        P[2, 1] = 0.1  # EX, MA
        P[2, 0] = 0.1
        P[0, 2] = 0.6  # MA, HE
        # P[0, 1] = 0.6;      P[1, 0] = 0.6   # HE, EX
        P[5, 4] = 0.6
        P[4, 5] = 0.2  # VOP, VHE
        P[0, 4] = -2  # 出血概念抑制

        # 共同特征逻辑
        for i in [5]:
            for j in [1, 2, 3]:
                P[i, j] = -0.6
        return P

    def forward(self, x):
        B, C, H, W = x.size()

        # # 1. 为 6 个通道手动设定不同的 tau 值
        # tau_values = [0.5, 0.5, 4.0, 3.0, 3.0, 8.0]

        # # 2. 转换为 Tensor，并调整形状为 [1, 6, 1, 1] 以触发广播机制
        # # 注意：一定要保持 dtype 和 device 与 x 一致，避免报错
        # tau = torch.tensor(tau_values, dtype=x.dtype, device=x.device).view(1, 6, 1, 1)

        # # 3. 基础映射：映射到 [0, 1] 概率空间
        # x_prob = torch.sigmoid(x / tau)
        # 假设 x 的形状是 [B, 6, H, W]，例如 [B, 6, 7, 7]
        tau = 0.5

        # --- 替换掉 self.norm_layer(x) 的部分 ---
        # 计算当前特征图在空间维度 (H, W) 上的均值和标准差
        # dim=(2, 3) 确保了每个样本(B)、每个通道(6)都是独立计算的
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5  # +1e-5 防止除以 0

        # 强行标准化：现在的 x_normalized 绝对是均值为 0，方差为 1 了
        x_normalized = (x - mean) / std
        # ---------------------------------------

        # 继续你的 Sigmoid
        x_prob = torch.sigmoid(x_normalized / tau)

        # 空间池化 [B, 6, 7, 7]
        # x_avg = F.adaptive_avg_pool2d(x_prob, (self.pool_size, self.pool_size))
        # x_max = F.adaptive_max_pool2d(x_prob, (self.pool_size, self.pool_size))
        # x_pooled = 0.3 * x_avg + 0.7 + x_max
        x_pooled = F.adaptive_avg_pool2d(x_prob, (self.pool_size, self.pool_size))

        # 展平为 [B, 6, 49] 以便进行单通道(单病灶)的空间统计
        x_spatial = x_pooled.view(B, self.num_concepts, -1)

        # ==================== 🌟 你的新思路：动态均值截断 ====================
        # 计算每张图、每种病灶的空间均值 [B, 6, 1]
        spatial_mean = x_spatial.mean(dim=-1, keepdim=True)

        # ==================== 🌟 核心修改：正负通道差异化剥离 ====================
        x_salient = torch.zeros_like(x_spatial)

        # 定义通道索引
        pos_idx = [0, 2, 4, 5]  # 正向响应通道：分数越高，病灶越强
        inv_idx = [1, 3]  # 倒置响应通道：分数越低，病灶越强
        # print(x_spatial)

        # A. 正向通道剥离 (保留高于均值的部分)
        x_salient[:, pos_idx, :] = F.relu(
            x_spatial[:, pos_idx, :] - spatial_mean[:, pos_idx, :]
        )
        # print(x_salient)
        # B. 倒置通道剥离与翻转 (保留低于均值的部分，并将其翻转为高亮)
        # 极其优雅的数学转换：既然越低越有病，我们就用 (均值 - 当前值)
        # 这样原本最低的值（最强病灶）会变成最大的正数！
        x_salient[:, inv_idx, :] = F.relu(
            spatial_mean[:, inv_idx, :] - x_spatial[:, inv_idx, :]
        )
        # ====================================================================

        # 此时的 x_salient 已经彻底统一了量纲：无论哪个通道，【值越大，就代表病灶越确定】
        x_flat_salient = x_salient.reshape(B, -1)

        # 此时的 x_gated 已经是：既剥离了空间底噪，又受全局患病概率控制的极度纯净的特征！
        x_gated = x_flat_salient.view(
            B, self.num_concepts, self.pool_size, self.pool_size
        )

        # 3. 图构建与归一化 + self.alpha * torch.tanh(self.P_delta)
        P_current = self.P_staticxx.to(x.device) + self.alpha * torch.tanh(self.P_delta)

        A = torch.kron(P_current, self.Dxx.to(x.device))

        # 4. 传递受控的干净信息
        message = torch.matmul(x_flat_salient, A)

        node_mixed = F.relu(message)

        # 6. 解码与上采样
        x_graph_spatial = node_mixed.view(
            B, self.num_concepts, self.pool_size, self.pool_size
        )

        # ==================== 🌟 核心还原：出图前翻转回倒置状态 ====================

        x_graph_spatial_out = x_graph_spatial.clone()
        # # 把原本是倒置的通道，再用 1.0 减去，恢复成“越低越是病灶”的原始状态！
        x_graph_spatial_out[:, inv_idx, :, :] = -x_graph_spatial[:, inv_idx, :, :]
        # print(x_graph_spatial_out)

        out = F.interpolate(
            x_graph_spatial_out, size=(H, W), mode="bilinear", align_corners=False
        )

        # 7. 提取图特征用于分类头
        g_mean = x_graph_spatial_out.mean(dim=(2, 3))
        g_max = x_graph_spatial_out.amax(dim=(2, 3))
        g_min = x_graph_spatial_out.amin(dim=(2, 3))

        graph_features_pooled = torch.cat([g_mean, g_max, g_min], dim=1)

        return graph_features_pooled, out
