import torch
import torch.nn as nn
import torch.nn.functional as F

class SignedGraphReasoning(nn.Module):
    """
    基于临床先验与高斯空间衰减的符号化动态图推理模块
    """
    def __init__(self, num_nodes=6, spatial_size=14, out_features=128, sigma=2.0, lambda_prior=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.spatial_size = spatial_size
        self.in_features = spatial_size * spatial_size # 展平后的维度 14*14 = 196
        self.out_features = out_features
        self.lambda_prior = lambda_prior

        # 1. 预计算高斯距离惩罚常数矩阵 (196 x 196)
        # 使用 register_buffer，这样它会被视为模型状态，自动跟随模型存入 GPU，但不参与梯度更新
        G = self._build_gaussian_kernel(spatial_size, sigma)
        self.register_buffer('G', G)

        # 2. 构建临床医学先验矩阵 (6 x 6)
        # 将其设置为 nn.Parameter，意味着以医学先验作为初始值，允许模型在训练中微调这个关系！
        P = self._build_prior_matrix()
        self.P = nn.Parameter(P)

        # 3. 图卷积节点特征更新层
        # 用于将融合后的 196 维特征映射为你需要的输出维度（如 128 或其他）
        self.W = nn.Linear(self.in_features, out_features)

    def _build_gaussian_kernel(self, size, sigma):
        """生成物理空间的二维高斯距离衰减矩阵"""
        coords = torch.arange(size, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        # 构建所有像素点的网格坐标 (196, 2)
        grid = torch.stack([y.flatten(), x.flatten()], dim=1)
        # 计算任意两点之间的欧式距离平方 (196, 196)
        dist_sq = torch.cdist(grid, grid, p=2)**2
        # 应用高斯衰减公式
        G = torch.exp(-dist_sq / (2 * sigma**2))
        return G

    def _build_prior_matrix(self):
        """
        手工设定先验连通性矩阵
        节点顺序约定: 0:HE(出血), 1:EX(硬性渗出), 2:MA(微动脉瘤), 3:SE(软性渗出), 4:VOP(玻璃体浑浊), 5:VHE(玻璃体出血)
        """
        P = torch.ones((6, 6)) * 0.1  # 基础微弱正相关
        P.fill_diagonal_(1.0)         # 自身到自身的权重为 1

        # 【正相关】：MA(2) 和 EX(1) (环形渗出包裹)
        P[2, 1] = P[1, 2] = 0.9
        # 【正相关】：MA(2) 和 HE(0) (破裂出血)
        P[2, 0] = P[0, 2] = 0.8
        # 【正相关】：EX(1) 和 SE(3) (中重度区域共现)
        P[1, 3] = P[3, 1] = 0.5
        # 【正相关】：VOP(4) 和 VHE(5)
        P[4, 5] = P[5, 4] = 0.8

        # 【强负相关】：VOP(4)及VHE(5) 遮挡 所有视网膜表面病灶(0, 1, 2, 3)
        for i in [4, 5]:
            for j in [0, 1, 2, 3]:
                P[i, j] = P[j, i] = -0.4

        return P

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.num_nodes

        # 展平特征 [B, 6, 196]
        x_flat = x.view(B, C, -1)

        # ==========================================
        # 🌟 修复 1：空间尺度正则化 (Spatial Min-Max Normalization)
        # 解决量纲爆炸问题，把所有特征强制拉平到 [0, 1] 区间
        # ==========================================
        x_min = x_flat.min(dim=-1, keepdim=True)[0]
        x_max = x_flat.max(dim=-1, keepdim=True)[0]
        # 加 1e-8 防止除以 0 导致 NaN
        x_norm = (x_flat - x_min) / (x_max - x_min + 1e-8)

        # ==========================================
        # 🌟 修复 2：反向特征的绝对语义翻转 (Semantic Alignment)
        # 解决语义偏移问题，让所有通道统一为 "越大代表病灶越严重"
        # ==========================================
        x_aligned = x_norm.clone()
        # VOP(4) 和 VHE(5) 是反向特征（原值越大越健康）
        # 1.0 - x 将它们翻转，现在 1.0 代表极度浑浊/出血，0.0 代表清澈！
        x_aligned[:, 1, :] = 1.0 - x_norm[:, 1, :]
        x_aligned[:, 3, :] = 1.0 - x_norm[:, 3, :]
        x_aligned[:, 5, :] = 1.0 - x_norm[:, 5, :]

        # === 重新计算空间共现矩阵 S ===
        # 使用对齐后的特征计算，并加上一个温度系数 5.0 让高激活区更锐利
        x_attn = F.softmax(x_aligned * 5.0, dim=-1)
        temp = torch.matmul(x_attn, self.G)
        S = torch.matmul(temp, x_attn.transpose(-2, -1))

        # === 生成最终符号邻接矩阵 A ===
        A = self.P * S + self.lambda_prior * self.P

        # ==========================================
        # 🌟 修复 3：安全的信息传递 (Message Passing)
        # 使用对齐且归一化后的 x_aligned 参与乘法，绝对不会梯度爆炸！
        # ==========================================
        node_mixed = torch.matmul(A, x_aligned)

        # 映射特征并使用 LeakyReLU 保留负向遮挡信号
        out_features = self.W(node_mixed)
        node_embeddings = F.leaky_relu(out_features, negative_slope=0.2)

        return node_embeddings, A
