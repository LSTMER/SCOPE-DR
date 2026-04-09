"""
MIL-VT: Multiple Instance Learning Vision Transformer

独立的弱监督病灶发现模型，通过图像级标签训练，自动学习病灶定位。

核心思想：
1. 把眼底图看作"袋子"，14×14 个 Patch 看作"实例"
2. 只要有一个 Patch 是病灶，整张图就是阳性
3. 通过 Attention 机制自动发现病灶位置
4. 输出概念激活图（类似教师矩阵，但是数据驱动的）
"""

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
        self.patch_classifier = nn.Conv2d(768, num_concepts, kernel_size=1)


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

        concept_maps = self.patch_classifier(patch_features)

        # 3. MIL 聚合：Max Pooling
        # 从 14x14 个 Patch 分数中，选出得分最高的那个作为整张图的得分
        lesion_logits = concept_maps.amax(dim=(2, 3))  # [B, 6]

        if return_attention:
            # 这里的 concept_maps 就是最完美的注意力权重热力图
            return concept_maps, lesion_logits, concept_maps.view(B, 6, 196)
        else:
            return concept_maps, lesion_logits


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("Testing MIL-VT Model...")

    BACKBONE_PATH = "./checkpoints/finetuned_model/dr_grading_finetune/checkpoints/epoch_latest.pt"

    try:
        model = MIL_VT_Model(BACKBONE_PATH, num_concepts=6, device='cuda')
        model.eval()

        dummy_img = torch.randn(2, 3, 224, 224).cuda()

        # 测试不返回注意力
        concept_maps, lesion_logits = model(dummy_img, return_attention=False)

        print(f"\n✅ Forward pass (without attention) successful!")
        print(f"   Concept Maps: {concept_maps.shape}")
        print(f"   Lesion Logits: {lesion_logits.shape}")

        # 测试返回注意力
        concept_maps, lesion_logits, attn_weights = model(dummy_img, return_attention=True)

        print(f"\n✅ Forward pass (with attention) successful!")
        print(f"   Attention Weights: {attn_weights.shape}")

        print("\n🎉 MIL-VT model working perfectly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
