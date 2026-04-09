import torch
import torch.nn as nn
import torch.nn.functional as F
from RET_CLIP.clip.utils import create_model


class MIL_VT_Model(nn.Module):
    """
    MIL-VT 完整模型

    输入: 眼底图像 [B, 3, 224, 224]
    输出:
        - concept_maps: 概念激活图 [B, 6, 14, 14]
        - grade_logits: 分级预测 [B, 5]
        - lesion_logits: 病灶预测 [B, 6]
        - attention_weights: 注意力权重 [B, 6, 196] (用于可视化)
    """
    def __init__(self, checkpoint_path, num_concepts=6, device='cuda'):
        super().__init__()
        self.device = device
        self.num_concepts = num_concepts

        # 1. 加载 Backbone (ViT-B/16)
        print(f"Loading backbone from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pass
        else:
            checkpoint = {'state_dict': checkpoint}

        self.clip_model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=checkpoint)
        self.clip_model.to(self.device)
        self.clip_model = self.clip_model.float()
        self.visual = self.clip_model.visual

        # 冻结 Backbone
        for param in self.visual.parameters():
            param.requires_grad = False

        # 2. 可学习的概念查询向量 [6, 768]
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, 768))
        nn.init.xavier_uniform_(self.concept_queries)

        # 3. Multi-Head Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )

        # 4. 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768 * 2, 768)
        )

        # 5. Layer Normalization
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)

        # 6. 空间注意力投影层：将概念特征投影到空间维度
        self.spatial_projector = nn.Linear(768, 196)

        # 7. 病灶分类头（MIL 聚合）
        # 使用 Max Pooling（MIL 的经典做法：只要有一个 Patch 是病灶，整张图就是阳性）
        self.lesion_head = nn.Sequential(
            nn.Linear(num_concepts, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_concepts)
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
            return_attention: 是否返回注意力权重

        Returns:
            concept_maps: [B, 6, 14, 14]
            grade_logits: [B, 5]
            lesion_logits: [B, 6]
            (可选) attention_weights: [B, 6, 196]
        """
        # 1. Backbone 提取 Patch Features
        with torch.no_grad():
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = torch.cat([
                self.visual.class_embedding.to(x.dtype) +
                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x
            ], dim=1)
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.visual.ln_post(x[:, 1:, :])

            B, L, D = x.shape
            H = W = int(L**0.5)  # 14
            patch_features = x.permute(0, 2, 1).reshape(B, D, H, W)  # [B, 768, 14, 14]

        # 2. 重塑为 [L, B, D] 用于 Attention
        patch_features_flat = patch_features.view(B, D, L).permute(2, 0, 1)  # [196, B, 768]

        patch_features_flat_b = patch_features_flat.permute(1, 0, 2)
        # queries: [B, 6, 768] (扩展为 batch)
        queries_b = self.concept_queries.unsqueeze(0).expand(B, -1, -1)

        # 4. 计算 Query 和每个 Patch 的相似度 (这才是真正的 Concept Map!)
        # [B, 6, 768] * [B, 768, 196] -> [B, 6, 196]
        similarity_scores = torch.bmm(queries_b, patch_features_flat_b.transpose(1, 2))

        # 为了梯度稳定，可以除以 sqrt(768)
        similarity_scores = similarity_scores / (768 ** 0.5)

        # 5. 生成空间概念图
        concept_maps = similarity_scores.view(B, self.num_concepts, H, W)  # [B, 6, 14, 14]

        # 6. MIL 聚合：Max Pooling
        lesion_logits = concept_maps.amax(dim=(2, 3))  # [B, 6]

        if return_attention:
            return concept_maps, lesion_logits, similarity_scores
        else:
            return concept_maps, lesion_logits
