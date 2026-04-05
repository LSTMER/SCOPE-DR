import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize

# === 导入你的模块 ===

from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from graph_model_cbm import SALF_CBM
from train_salf_cbm_end2end import SmartFundusCrop, Config # 复用之前的配置

class SALF_Experiment_Wrapper(nn.Module):
    def __init__(self, pretrained_salf_model, experiment_mode='concat_mlp', freeze_base=True):
        """
        Args:
            pretrained_salf_model: 已经训练好的 SALF_CBM 实例
            experiment_mode: 选择当前测试的模块组合模式
            freeze_base: 是否冻结底层已训练好的特征提取器参数
        """
        super().__init__()
        # 1. 挂载已训练的底层模型
        self.base = pretrained_salf_model

        # 2. 冻结底层参数 (用于单独测试新组合的模块，防止破坏已学到的概念映射和图结构)
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.experiment_mode = experiment_mode

        # 获取底层维度信息
        self.num_concepts = self.base.num_concepts
        self.concept_dim = self.num_concepts * 3     # 18维 (mean, max, min)
        self.graph_dim = self.num_concepts * 3       # 18维 (根据你的 SpatialConceptGraph 实际返回)
        self.global_dim = 768                        # ViT 输出维度

        # ====================================================================
        # ★★★ 在这里定义你想测试的不同网络层组合 (Experiment Heads) ★★★
        # ====================================================================

        if self.experiment_mode == 'baseline_reproduce':
            # 模式 A: 仅重现你原有的最终层逻辑，用于对照测试
            self.grade_head = nn.Linear(self.concept_dim + self.graph_dim, 5)
            self.lesion_head = nn.Sequential(
                nn.Linear(self.concept_dim + self.graph_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_concepts)
            )

        elif self.experiment_mode == 'global_fusion_mlp':
            # 模式 B: 新增层 - 将图特征(18)、概念特征(18)与全局池化特征(768)全面拼接，经过深层MLP
            in_dim = self.concept_dim + self.graph_dim + self.global_dim
            self.grade_head = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )
            self.lesion_head = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_concepts)
            )

        elif self.experiment_mode == 'cross_attention_fusion':
            # 模式 C: 新增层 - 使用自注意力机制来融合特征
            # 将 global, concept, graph 投影到相同维度后进行注意力交互
            self.common_dim = 64
            self.proj_global = nn.Linear(self.global_dim, self.common_dim)
            self.proj_concept = nn.Linear(self.concept_dim, self.common_dim)
            self.proj_graph = nn.Linear(self.graph_dim, self.common_dim)

            # Transformer 编码层用于特征融合
            self.fusion_encoder = nn.TransformerEncoderLayer(
                d_model=self.common_dim, nhead=4, batch_first=True
            )

            self.grade_head = nn.Linear(self.common_dim, 5)
            self.lesion_head = nn.Linear(self.common_dim, self.num_concepts)

    def forward(self, x):
        # ====================================================================
        # 1. 提取特征：复用原 SALF_CBM 的前向逻辑，但我们只截取特征
        # ====================================================================
        with torch.no_grad() if not self.base.training else torch.enable_grad():
            # 提取 ViT 特征
            feat = self.base.visual.conv1(x)
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1).permute(0, 2, 1)
            feat = torch.cat([self.base.visual.class_embedding.to(feat.dtype) + torch.zeros(feat.shape[0], 1, feat.shape[-1], dtype=feat.dtype, device=feat.device), feat], dim=1)
            feat = feat + self.base.visual.positional_embedding.to(feat.dtype)
            feat = self.base.visual.ln_pre(feat)
            feat = feat.permute(1, 0, 2)
            feat = self.base.visual.transformer(feat)
            feat = feat.permute(1, 0, 2)
            feat = self.base.visual.ln_post(feat[:, 1:, :])

            B, L, D = feat.shape
            H = W = int(L**0.5)
            img_feats = feat.permute(0, 2, 1).reshape(B, D, H, W) # [B, 768, 14, 14]

            # 提取全局特征
            global_features = img_feats.mean(dim=(2, 3)) # [B, 768]

            # 提取概念特征
            concept_maps = self.base.concept_projector(img_feats)
            c_mean = concept_maps.mean(dim=(2, 3))
            c_max = concept_maps.amax(dim=(2, 3))
            c_min = concept_maps.amin(dim=(2, 3))
            concept_features = torch.cat([c_mean, c_max, c_min], dim=1) # [B, 18]

            # 获取原模型的辅助病灶预测 (用于图网络的输入)
            lesion_logits_aux = self.base.aux_head(concept_features)

            # 提取图推理特征
            graph_features, after_graph_map = self.base.spatial_graph(concept_maps, lesion_logits_aux) # [B, 18]

        # ====================================================================
        # 2. 实验性模块分支：将截取到的特征送入你新添加的测试层
        # ====================================================================

        if self.experiment_mode == 'baseline_reproduce':
            fused_features = torch.cat([concept_features, graph_features], dim=1)
            grade_out = self.grade_head(fused_features)
            lesion_out = self.lesion_head(fused_features)

        elif self.experiment_mode == 'global_fusion_mlp':
            # 将所有特征拼接：18 + 18 + 768 = 804 维
            all_features = torch.cat([concept_features, graph_features, global_features], dim=1)
            grade_out = self.grade_head(all_features)
            lesion_out = self.lesion_head(all_features)

        elif self.experiment_mode == 'cross_attention_fusion':
            # 将特征映射到相同维度 [B, 1, 64]
            g_feat = self.proj_global(global_features).unsqueeze(1)
            c_feat = self.proj_concept(concept_features).unsqueeze(1)
            gr_feat = self.proj_graph(graph_features).unsqueeze(1)

            # 序列化并进行注意力融合 [B, 3, 64]
            seq_features = torch.cat([g_feat, c_feat, gr_feat], dim=1)
            fused_seq = self.fusion_encoder(seq_features)

            # 取平均池化作为最终表达 [B, 64]
            final_repr = fused_seq.mean(dim=1)
            grade_out = self.grade_head(final_repr)
            lesion_out = self.lesion_head(final_repr)

        else:
            raise ValueError(f"Unknown experiment mode: {self.experiment_mode}")

        # 你依然可以返回可见的 concept_maps 和 after_graph_map 用于可视化
        return grade_out, lesion_out, concept_maps, after_graph_map

from evaluate_cbm import EvalConfig

cfg = EvalConfig()
# 1. 初始化你原本的 SALF 模型并加载权重
concepts = ["HE", "EX", "MA", "SE", "VHE", "VOP"]
model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE)
model.to(cfg.DEVICE)

checkpoint = torch.load(cfg.CHECKPOINT_PATH6, map_location=cfg.DEVICE)
model.load_state_dict(checkpoint, strict=False)
print("✅ Weights successfully loaded!")

# 假设你已经有训练好的 SALF_CBM 的完整权重，也可以在这里 load
# original_salf.load_state_dict(torch.load("trained_salf_cbm.pth"))

# 2. 包装模型，选择你想测试的层组合
# 模式: 'baseline_reproduce', 'global_fusion_mlp', 'cross_attention_fusion'
experiment_model = SALF_Experiment_Wrapper(
    pretrained_salf_model=model,
    experiment_mode='global_fusion_mlp',  # <--- 随时切换你想测试的架构
    freeze_base=True  # 设置为True则只训练新的 MLP/Attention 头
).to('cuda')

# 3. 正常送入优化器 (注意这里只会优化 require_grad=True 的参数，即你新加的层)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, experiment_model.parameters()), lr=1e-4)

# 4. 前向传播
images = torch.randn(2, 3, 224, 224).to('cuda')
grade_out, lesion_out, concept_maps, after_graph_map = experiment_model(images)
