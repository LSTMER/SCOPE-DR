import torch
import torch.nn as nn
import torch.nn.functional as F


class MIL_VT_Projector(nn.Module):
    """
    MIL-VT 投影器：基于 Cross-Attention 的数据驱动病灶发现模块

    核心思想：
    - 为每个概念学习一个可查询向量（Concept Query）
    - 通过 Cross-Attention 从 Patch Features 中自主发掘病灶区域
    - 输出与 CLIP_Projector 相同尺寸的概念图 [B, 6, 14, 14]
    """

    def __init__(self, num_concepts=6, feature_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # 1. 可学习的概念查询向量 [6, 768]
        # 每个概念对应一个查询向量，用于在 Patch 中搜索相关病灶
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, feature_dim))
        nn.init.xavier_uniform_(self.concept_queries)

        # 2. Multi-Head Cross-Attention
        # Query: Concept Queries [6, 768]
        # Key/Value: Patch Features [196, 768]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # 使用 (L, B, D) 格式
        )

        # 3. 前馈网络（可选，用于增强表达能力）
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # 4. Layer Normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # 5. 空间注意力投影层：将 [6, 768] 的概念特征投影到 [6, 196] 的空间权重
        self.spatial_projector = nn.Linear(feature_dim, 196)

    def forward(self, patch_features):
        """
        Args:
            patch_features: [B, 768, 14, 14] - ViT 输出的 Patch Features

        Returns:
            concept_maps: [B, 6, 14, 14] - 数据驱动的概念激活图
            attention_weights: [B, 6, 196] - 每个概念对每个 Patch 的注意力权重（用于可视化）
        """
        B, C, H, W = patch_features.shape
        L = H * W  # 196

        # 1. 重塑 Patch Features: [B, 768, 14, 14] -> [196, B, 768]
        patch_features_flat = patch_features.view(B, C, L).permute(2, 0, 1)  # [L, B, D]

        # 2. 扩展 Concept Queries: [6, 768] -> [6, B, 768]
        queries = self.concept_queries.unsqueeze(1).expand(-1, B, -1)  # [6, B, 768]

        # 3. Cross-Attention: Concept Queries 查询 Patch Features
        # attn_output: [6, B, 768] - 每个概念聚合后的特征
        # attn_weights: [B, 6, 196] - 注意力权重矩阵
        attn_output, attn_weights = self.cross_attention(
            query=queries,  # [6, B, 768]
            key=patch_features_flat,  # [196, B, 768]
            value=patch_features_flat,
            need_weights=True,
            average_attn_weights=True,  # 对多头取平均
        )

        # 4. 残差连接 + Layer Norm
        attn_output = self.norm1(attn_output + queries)  # [6, B, 768]

        # 5. 前馈网络
        ffn_output = self.ffn(attn_output)
        concept_features = self.norm2(ffn_output + attn_output)  # [6, B, 768]

        # 6. 生成空间概念图
        # 方法：将概念特征投影到空间维度，得到每个位置的激活强度
        concept_features = concept_features.permute(1, 0, 2)  # [B, 6, 768]

        # 投影到空间维度: [B, 6, 768] -> [B, 6, 196]
        spatial_activations = self.spatial_projector(concept_features)  # [B, 6, 196]

        # 重塑为空间图: [B, 6, 196] -> [B, 6, 14, 14]
        concept_maps = spatial_activations.view(B, self.num_concepts, H, W)

        # 7. 应用 Sigmoid 激活，将值映射到 [0, 1]
        concept_maps = torch.sigmoid(concept_maps)

        return concept_maps, attn_weights


import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionModule(nn.Module):
    def __init__(self, num_concepts=6):
        super().__init__()
        self.num_concepts = num_concepts
        self.gate_network = nn.Sequential(
            nn.Linear(num_concepts * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_concepts),
            nn.Sigmoid(),
        )

    def global_min_max_norm(self, x, eps=1e-8):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, -1)
        x_min = x_flat.min(dim=-1, keepdim=True)[0].view(B, 1, 1, 1)
        x_max = x_flat.max(dim=-1, keepdim=True)[0].view(B, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, map_clip, map_trans):
        # 1. 全局极值尺度对齐
        norm_clip = self.global_min_max_norm(map_clip)
        norm_trans = self.global_min_max_norm(map_trans)

        # 2. 基础全局门控融合
        B = norm_clip.size(0)
        feat_clip = F.adaptive_avg_pool2d(norm_clip, (1, 1)).view(B, self.num_concepts)
        feat_trans = F.adaptive_avg_pool2d(norm_trans, (1, 1)).view(
            B, self.num_concepts
        )
        feat_combined = torch.cat([feat_clip, feat_trans], dim=1)
        alpha = self.gate_network(feat_combined).view(B, self.num_concepts, 1, 1)

        base_fused_maps = alpha * norm_clip + (1.0 - alpha) * norm_trans

        # ==========================================
        # 🌟 核心升级：双重约束稀疏软路由 (Dual-Constraint Sparse Routing)
        # ==========================================
        # 条件 1: 全局均值 (跨通道、跨空间) -> 绝对门槛
        global_mean = norm_trans.mean(dim=(1, 2, 3), keepdim=True) * 0.8  # [B, 1, 1, 1]

        # 条件 2: 局部通道均值 (仅跨空间) -> 相对门槛
        local_mean = norm_trans.mean(dim=(2, 3), keepdim=True) * 1.2  # [B, 6, 1, 1]

        # 计算两个边界的差值（Margin）
        margin_global = norm_trans - global_mean
        margin_local = norm_trans - local_mean

        # 💡 极其优雅的“软交集”：取两者的最小值！
        # 只有当像素同时大于全局均值和局部均值时，strict_margin 才会是正数
        # 只要有一个不满足，strict_margin 就是负数，掩码就会趋向于 0
        strict_margin = torch.minimum(margin_global, margin_local)

        # 通过 Sigmoid 将差异转化为 [0, 1] 的平滑掩码
        tau = 0.05  # 温度系数，调小一点让掩码更锐利、选取的极值更少
        soft_mask = torch.sigmoid(strict_margin / tau)

        # 最终融合：极少数的、最尖锐的病灶点用 MIL (norm_trans)，绝大部分用 base_fused_maps
        final_fused_maps = soft_mask * norm_trans + (1.0 - soft_mask) * base_fused_maps

        return final_fused_maps, alpha
