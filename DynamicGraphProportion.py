import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialConceptGraph(nn.Module):
    def __init__(self, num_concepts=6, pool_size=7, sigma=0.8, out_features=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.pool_size = pool_size
        self.num_spatial_nodes = pool_size * pool_size
        self.total_nodes = num_concepts * self.num_spatial_nodes

        # 1. 静态空间距离衰减矩阵 D (不参与学习)
        D_X = self._build_spatial_distance(pool_size, sigma)
        # pool_size = 7
        # self.pool_size = 7
        # self.D_X = self._build_spatial_distance(pool_size, sigma)
        self.register_buffer('D_X', D_X)

        # 2. 静态医学先验矩阵 P_static (不参与学习)
        P_staticx = self._build_prior_matrix()
        self.register_buffer('P_staticx', P_staticx)

        self.P_static = self._build_prior_matrix()

        # 3. 半静态学习的核心：可学习的微调矩阵 P_delta
        # 初始化为全0，意味着训练初期完全依靠专家的 P_static
        self.P_delta = nn.Parameter(torch.zeros(num_concepts, num_concepts))

        # 微调幅度锁 (控制模型最多偏离专家经验多少，0.2是一个适中的值)
        self.alpha = 0.2

        # self.proj = nn.Linear(self.total_nodes, out_features)

    def _build_spatial_distance(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([y.flatten(), x.flatten()], dim=1)
        dist_sq = torch.cdist(grid, grid, p=2)**2
        return torch.exp(-dist_sq / (2 * sigma**2))

    def _build_prior_matrix(self):
        """医学先验连通性矩阵"""
        """ 0:HE(出血), 1:EX(硬性渗出), 2:MA(微动脉瘤), 3:SE(软性渗出), 4:VHE(玻璃体出血), 5: VOP(玻璃体浑浊)
            其中, HE, EX, SE 是负相关的通道 因此其传递给正相关通道时的应该是负数"""
        P = torch.zeros((6, 6))
        P[0, 0] = 1
        P[1, 1] = 1
        P[2, 2] = 0.2
        P[3, 3] = 1
        P[4, 4] = 0.5
        P[5, 5] = 1

        P[1, 2] = -0.9
        P[2, 1] = 0.1

        P[2, 0] = 0.1
        P[0, 2] = -0.8

        P[0, 1] = 0.6
        P[1, 0] = 0.6

        P[5, 4] = 1
        P[4, 5] = 0.1

        P[5, 2] = -0.6

        for i in [5]:
            for j in [0, 1, 3]:
                P[i, j] = 0.6
        return P

    import torch
    import torch.nn.functional as F

    def forward(self, x, lesion_logit):
        # 1. 空间池化与展平 [B, 96]
        B, C, H, W = x.size()
        x_pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        x_flat = x_pooled.reshape(B, -1)

        confidence = torch.sigmoid(lesion_logit) # [B, 6]
        c_nodes = confidence.unsqueeze(-1).expand(-1, -1, self.num_spatial_nodes).reshape(B, -1)

        # ==================== 核心修改区：自身特征门控 ====================

        # 2. 计算当前 forward 步的动态语义矩阵 P
        P_current = self.P_static.to(x.device) + self.alpha * torch.tanh(self.P_delta)

        # D = self.D_X.to(x.device)

        # 3. 极速生成 96x96 的终极联合邻接矩阵 A
        A = torch.kron(P_current, self.D_X)

        degree = torch.sum(torch.abs(A), dim=1, keepdim=True) + 1e-8

        # 归一化邻接矩阵 A -> A_norm
        # 现在 A_norm 中每一行的绝对值加起来刚好等于 1.0！
        A_norm = A / degree

        # 3. 极简的图信息传递 (真正的加权平均！)
        # 因为 A_norm 已经包含了对角线(自身)且归一化了，所以出来的直接就是完美平衡的特征
        node_mixed = torch.matmul(x_flat, A_norm)

        # print(P_current)

        # 🌟 核心改进：三次方自门控 (Self-Gating)
        # 只有响应足够强烈（激活度高）的节点，才有资格大声说话
        # 极其微小的响应（如0.1~0.3）被三次方后几乎归零，防止了信息洪流污染背景
        # 结合 Tanh 将广播能量软性锁死在 [-1, 1] 之间，同时保留三次方压制小数值的特性
        # x_broadcast = x_flat * torch.pow(torch.tanh(x_flat), 2) * c_nodes

        # # 4. 图信息传递 (Message Passing) - 传递的是门控过滤后的信息
        # influence = torch.matmul(x_broadcast, A)

        # ====================================================================

        # # 5. 动态保留机制 (保护原图自身特征)
        # # 根据外来影响的大小，决定坚守多少自我
        # alpha_keep = torch.clamp(1.0 / (torch.abs(influence) + 1e-5), max=1.0)
        # node_mixed = influence

        # ================= 6. 解码与池化融合 =================
        # 解包回 6x4x4
        x_graph_spatial = node_mixed.view(B, self.num_concepts, self.pool_size, self.pool_size)
        # print(x_graph_spatial)

        # 直接双线性插值上采样到 6x14x14
        x_upsampled = F.interpolate(x_graph_spatial, size=(H, W), mode='bilinear', align_corners=False)

        # 简单粗暴叠加！你加的 * 0.1 非常好，作为残差缩放防止图特征喧宾夺主
        out = x_upsampled

        # 全局统计池化 (提取用于分类头的 18 维特征)
        g_mean = x_graph_spatial.mean(dim=(2, 3))  # [B, 6]
        g_max = x_graph_spatial.amax(dim=(2, 3))   # [B, 6]
        g_min = x_graph_spatial.amin(dim=(2, 3))   # [B, 6]

        graph_features_pooled = torch.cat([g_mean, g_max, g_min], dim=1) # [B, 18]

        return graph_features_pooled, out
