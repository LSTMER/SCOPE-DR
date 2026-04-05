import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialConceptGraph(nn.Module):
    """
    优化的空间-语义联合图推理模块
    将输入图池化为小尺寸(如4x4)，对每个空间网格进行跨概念的图消息传递
    """
    def __init__(self, num_concepts=6, pool_size=4, sigma=1.0, out_features=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.pool_size = pool_size
        self.num_spatial_nodes = pool_size * pool_size  # 例如 4x4 = 16
        self.total_nodes = num_concepts * self.num_spatial_nodes # 例如 6x16 = 96

        # 1. 预计算 16x16 的空间距离衰减矩阵 D
        D = self._build_spatial_distance(pool_size, sigma)

        # 2. 获取 6x6 的医学先验矩阵 P
        P = self._build_prior_matrix()

        # 3. 构建 96x96 的终极联合邻接矩阵 A
        # A[i, j] 包含了概念先验和空间距离的双重约束
        A = self._build_joint_adjacency(P, D)
        self.register_buffer('A', A) # 作为静态矩阵存入模型，暂时不参与训练

        # 4. 降维映射层 (将 96 维的图特征压缩，防止在拼接时喧宾夺主)
        self.proj = nn.Linear(self.total_nodes, out_features)

    def _build_spatial_distance(self, size, sigma):
        """生成 16x16 的空间高斯距离矩阵"""
        coords = torch.arange(size, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([y.flatten(), x.flatten()], dim=1)
        dist_sq = torch.cdist(grid, grid, p=2)**2
        return torch.exp(-dist_sq / (2 * sigma**2))

    def _build_prior_matrix(self):
        """医学先验连通性矩阵 (0:HE, 1:EX, 2:MA, 3:SE, 4:VOP, 5:VHE)"""
        P = torch.ones((6, 6)) * 0.1
        P.fill_diagonal_(1.0)
        # 正相关示例
        P[2, 1] = P[1, 2] = 0.9  # MA & EX
        P[2, 0] = P[0, 2] = 0.8  # MA & HE
        P[4, 5] = P[5, 4] = 0.8  # VOP & VHE
        # 负向遮挡示例
        for i in [4, 5]:
            for j in [0, 1, 2, 3]:
                P[i, j] = -0.8
        return P

    def _build_joint_adjacency(self, P, D):
        """
        核心逻辑：组合空间与先验
        对于任意概念 c1 的位置 p1，与概念 c2 的位置 p2 之间的关联强度为：P[c1, c2] * D[p1, p2]
        """
        A = torch.zeros(self.num_concepts, self.num_spatial_nodes,
                        self.num_concepts, self.num_spatial_nodes)

        for c1 in range(self.num_concepts):
            for c2 in range(self.num_concepts):
                # 如果这两个病灶在医学上正相关，且当前这两个节点在空间上很近(D很大)
                # 那么它们之间的边权重就会非常大！
                A[c1, :, c2, :] = P[c1, c2] * D

        # 展平为 [96, 96] 的矩阵
        return A.view(self.total_nodes, self.total_nodes)

    def forward(self, x):
        """
        x: [B, 6, 14, 14] 的概念热力图
        """
        # 1. 空间池化降维 (14x14 -> 4x4)
        x_pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size)) # [B, 6, 4, 4]

        # 2. 展平为节点特征向量 [B, 96]
        x_flat = x_pooled.reshape(x.size(0), -1)

        # 3. 静态图信息传递 (Message Passing) 与 动态保留机制
        # 3.1 获取外来的影响信息 (Message)
        influence = torch.matmul(x_flat, self.A.t()) # [B, 96]

        # 3.2 计算动态保留系数 alpha (限制最大值为1.0，加上1e-5防止除零)
        # 取绝对值是因为负数的影响(强抑制)同样也是一种强影响
        alpha = torch.clamp(1.0 / (torch.abs(influence) + 1e-5), max=1.0)

        # 3.3 融合：(动态缩放的原特征) + (接收到的影响信息)
        node_mixed = alpha * x_flat + influence

        # 4. 激活与降维压缩
        node_activated = F.leaky_relu(node_mixed, negative_slope=0.2)
        graph_features = self.proj(node_activated) # [B, 64]

        return graph_features
