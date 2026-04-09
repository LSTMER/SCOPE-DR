"""
SALF-CBM Fusion Model (简化版)

架构：
1. CLIP Projector: 已训练好，加载权重
2. MIL-VT: 独立训练好，加载完整模型
3. Gated Fusion: 融合两条路的概念图
4. Graph + Heads: 后续推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from RET_CLIP.clip.utils import create_model
from DynamicGraphProportion import SpatialConceptGraph
from mil_vt_modules import GatedFusionModule
from mil_vt_model import MIL_VT_Model


class SALF_CBM_Fusion(nn.Module):
    def __init__(self, checkpoint_path, concepts, mil_vt_checkpoint=None, device='cuda'):
        """
        Args:
            checkpoint_path: RET-CLIP backbone 权重
            concepts: 概念列表
            clip_checkpoint: CLIP 投影器权重（可选）
            mil_vt_checkpoint: MIL-VT 完整模型权重（可选）
            device: 设备
        """
        super().__init__()
        self.device = device
        self.concepts = concepts
        self.num_concepts = len(concepts)

        # ==========================================
        # 1. 加载 Backbone (ViT-B/16)
        # ==========================================
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

        # ==========================================
        # 2. CLIP 先验投影器
        # ==========================================
        self.concept_projector = nn.Conv2d(
            in_channels=768,
            out_channels=self.num_concepts,
            kernel_size=1,
            bias=False
        ).to(self.device)

        # 用文本特征初始化
        self._initialize_concept_projector_with_text()

        # # 如果提供了 CLIP 权重，加载它
        # if clip_checkpoint is not None:
        #     self.load_clip_weights(clip_checkpoint)

        # ==========================================
        # 3. MIL-VT 模型（独立训练好的）
        # ==========================================
        print("Initializing MIL-VT model...")
        self.mil_vt = MIL_VT_Model(
            checkpoint_path=checkpoint_path,
            num_concepts=self.num_concepts,
            device=self.device
        ).to(self.device)

        # 如果提供了 MIL-VT 权重，加载它
        if mil_vt_checkpoint is not None:
            self.load_mil_vt_weights(mil_vt_checkpoint)

        # 冻结 MIL-VT（已经训练好）
        for param in self.mil_vt.parameters():
            param.requires_grad = False

        # ==========================================
        # 4. 门控融合模块
        # ==========================================
        self.fusion_module = GatedFusionModule(
            num_concepts=self.num_concepts
        ).to(self.device)

        # ==========================================
        # 5. 图推理模块
        # ==========================================
        self.graph_output = 18
        self.spatial_graph = SpatialConceptGraph(
            num_concepts=self.num_concepts,
            pool_size=7
        ).to(device)

        # ==========================================
        # 6. 分类头
        # ==========================================
        concept_dim = self.num_concepts * 3  # 18
        fused_dim = concept_dim + self.graph_output  # 36

        self.headx = nn.Linear(concept_dim, 5).to(device)

        self.aux_head = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_concepts)
        ).to(device)

        self.lesion_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_concepts)
        ).to(device)

        self.final_headx = nn.Linear(fused_dim, 5).to(device)

    def _initialize_concept_projector_with_text(self):
        """使用 CLIP 文本特征初始化先验投影器"""
        print("Initializing CLIP projector with text embeddings...")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")

        prompts = [f"一张包含{c}的眼底照片" for c in self.concepts]
        text_inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            text_output = self.clip_model.encode_text(text_inputs['input_ids'])
            text_feats = text_output[0]
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

            visual_proj = self.visual.proj.to(self.device)
            fused_weight = text_feats @ visual_proj.t()

            self.concept_projector.weight.data = fused_weight.view(self.num_concepts, 768, 1, 1)
            self.concept_projector.weight.requires_grad = True

    def load_clip_weights(self, clip_checkpoint_path):
        """加载预训练的 CLIP 投影器权重"""
        print(f"Loading CLIP projector weights from {clip_checkpoint_path}...")
        checkpoint = torch.load(clip_checkpoint_path, map_location=self.device)
        # 提取 concept_projector 的权重
        clip_weights = {k.replace('concept_projector.', ''): v for k, v in checkpoint.items() if 'concept_projector' in k}
        if clip_weights:
            self.concept_projector.load_state_dict(clip_weights, strict=False)
            print("✅ CLIP projector weights loaded!")
        else:
            print("⚠️ No CLIP projector weights found in checkpoint")

    def load_mil_vt_weights(self, mil_vt_checkpoint_path):
        """加载预训练的 MIL-VT 权重"""
        print(f"Loading MIL-VT weights from {mil_vt_checkpoint_path}...")
        checkpoint = torch.load(mil_vt_checkpoint_path, map_location=self.device)
        self.mil_vt.load_state_dict(checkpoint)
        print("✅ MIL-VT weights loaded!")

    def forward(self, x, return_attention=False):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
            return_attention: 是否返回 MIL-VT 的注意力权重

        Returns:
            grade_logits: 初步分级 logits [B, 5]
            fused_concept_maps: 融合后的概念图 [B, 6, 14, 14]
            lesion_logits_aux: 辅助病灶分类 logits [B, 6]
            lesion_logits_graph: 图推理病灶分类 logits [B, 6]
            grade_logits_final: 最终分级 logits [B, 5]
            after_graph_map: 图推理后的概念图 [B, 6, 14, 14]
            (可选) attention_weights: MIL-VT 注意力权重 [B, 6, 196]
            (可选) gate_alpha: 门控权重 [B, 6]
        """
        # ==========================================
        # 1. Backbone 前向传播
        # ==========================================
        with torch.no_grad():
            feat = self.visual.conv1(x)
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1).permute(0, 2, 1)
            feat = torch.cat([
                self.visual.class_embedding.to(feat.dtype) +
                torch.zeros(feat.shape[0], 1, feat.shape[-1], dtype=feat.dtype, device=feat.device),
                feat
            ], dim=1)
            feat = feat + self.visual.positional_embedding.to(feat.dtype)
            feat = self.visual.ln_pre(feat)
            feat = feat.permute(1, 0, 2)
            feat = self.visual.transformer(feat)
            feat = feat.permute(1, 0, 2)
            feat = self.visual.ln_post(feat[:, 1:, :])

            B, L, D = feat.shape
            H = W = int(L**0.5)
            patch_features = feat.permute(0, 2, 1).reshape(B, D, H, W)  # [B, 768, 14, 14]

        # ==========================================
        # 2. 双路投影：CLIP 先验 + MIL-VT 数据驱动
        # ==========================================
        # A. CLIP 先验路
        clip_concept_maps = self.concept_projector(patch_features)  # [B, 6, 14, 14]

        # B. MIL-VT 数据驱动路（调用预训练的 MIL-VT）
        with torch.no_grad():
            if return_attention:
                mil_concept_maps, _, attention_weights = self.mil_vt(x, return_attention=True)
            else:
                mil_concept_maps, _ = self.mil_vt(x, return_attention=False)
                attention_weights = None

        # ==========================================
        # 3. 门控融合
        # ==========================================
        # fused_concept_maps, gate_alpha = self.fusion_module(clip_concept_maps, mil_concept_maps)
        # print(gate_alpha)

        # ==========================================
        # 4. 概念特征提取
        # ==========================================
        # c_mean = fused_concept_maps.mean(dim=(2, 3))
        # c_max = fused_concept_maps.amax(dim=(2, 3))
        # c_min = fused_concept_maps.amin(dim=(2, 3))
        # concept_features = torch.cat([c_mean, c_max, c_min], dim=1)  # [B, 18]

        c_mean = clip_concept_maps.mean(dim=(2, 3))
        c_max = clip_concept_maps.amax(dim=(2, 3))
        c_min = clip_concept_maps.amin(dim=(2, 3))
        concept_features = torch.cat([c_mean, c_max, c_min], dim=1)  # [B, 18]

        # ==========================================
        # 5. 辅助病灶分类
        # ==========================================
        lesion_logits_aux = self.aux_head(concept_features)

        # ==========================================
        # 6. 图推理
        # ==========================================
        # graph_features, after_graph_map = self.spatial_graph(
        #     fused_concept_maps
        # )
        graph_features, after_graph_map = self.spatial_graph(
            clip_concept_maps
        )
        # 图推理后的结果和mil进行门控融合
        fused_concept_maps, gate_alpha = self.fusion_module(after_graph_map, mil_concept_maps)
        # print(gate_alpha)

        f_mean = fused_concept_maps.mean(dim=(2, 3))
        f_max = fused_concept_maps.amax(dim=(2, 3))
        f_min = fused_concept_maps.amin(dim=(2, 3))
        stat_features = torch.cat([f_mean, f_max, f_min], dim=1) # [B, 18]

        # ==========================================
        # 7. 特征融合与最终分类
        # ==========================================
        fused_features = torch.cat([concept_features, stat_features], dim=1)  # [B, 36]

        grade_logits = self.headx(concept_features)
        lesion_logits_graph = self.lesion_head(fused_features)
        grade_logits_final = self.final_headx(fused_features)

        # ==========================================
        # 8. 返回结果
        # ==========================================
        if return_attention:
            return (
                grade_logits,
                fused_concept_maps,
                lesion_logits_aux,
                lesion_logits_graph,
                grade_logits_final,
                after_graph_map,
                attention_weights,
                gate_alpha
            )
        else:
            return (
                grade_logits,
                mil_concept_maps,
                lesion_logits_aux,
                lesion_logits_graph,
                grade_logits_final,
                fused_concept_maps
            )


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("Testing SALF_CBM_Fusion Model (V2)...")

    CONCEPTS = ["HE", "EX", "MA", "SE", "VHE", "VOP"]
    BACKBONE_PATH = "./checkpoints/finetuned_model/dr_grading_finetune/checkpoints/epoch_latest.pt"

    try:
        model = SALF_CBM_Fusion(
            checkpoint_path=BACKBONE_PATH,
            concepts=CONCEPTS,
            device='cuda'
        )
        model.eval()

        dummy_img = torch.randn(2, 3, 224, 224).cuda()

        # 测试不返回注意力
        outputs = model(dummy_img, return_attention=False)
        print(f"\n✅ Forward pass (without attention) successful!")
        print(f"   Grade Logits: {outputs[0].shape}")
        print(f"   Fused Concept Maps: {outputs[1].shape}")
        print(f"   Lesion Logits (Aux): {outputs[2].shape}")
        print(f"   Lesion Logits (Graph): {outputs[3].shape}")
        print(f"   Grade Logits (Final): {outputs[4].shape}")
        print(f"   After Graph Map: {outputs[5].shape}")

        # 测试返回注意力
        outputs_with_attn = model(dummy_img, return_attention=True)
        print(f"\n✅ Forward pass (with attention) successful!")
        print(f"   Attention Weights: {outputs_with_attn[6].shape}")
        print(f"   Gate Alpha: {outputs_with_attn[7].shape}")

        print("\n🎉 Fusion model (V2) working perfectly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
