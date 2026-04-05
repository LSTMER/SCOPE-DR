import torch
import torch.nn as nn
from RET_CLIP.clip.utils import create_model
from DynamicGraphProportion import SpatialConceptGraph

class SALF_CBM(nn.Module):
    def __init__(self, checkpoint_path, concepts, device='cuda'):
        """
        Args:
            checkpoint_path: 你微调好的 RET-CLIP 权重路径
            concepts: 概念列表 (List of strings)
        """
        super().__init__()
        self.device = device
        self.concepts = concepts
        self.num_concepts = len(concepts)

        # 1. 加载 Backbone (ViT-B/16)
        # 注意：我们需要魔改一下 CLIP，让它返回 Feature Map 而不是 CLS token
        print(f"Loading backbone from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint: pass
        else: checkpoint = {'state_dict': checkpoint}

        # 加载完整模型
        self.clip_model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=checkpoint)
        self.clip_model.to(self.device)
        self.clip_model = self.clip_model.float()
        self.visual = self.clip_model.visual # 只取视觉部分

        # 2. 冻结 Backbone (因为已经微调很好了)
        for param in self.visual.parameters():
            param.requires_grad = False

        # 3. [核心模块] 概念映射层 (The Concept Bottleneck Layer)
        # 本质上是一个将 768维 -> C维 的投影层
        # 我们用 nn.Conv2d(1x1) 来实现
        self.concept_projector = nn.Conv2d(in_channels=768, out_channels=self.num_concepts, kernel_size=1, bias=False).to(self.device)

        # ★★★ 关键步骤：用文本特征初始化投影层 ★★★
        self._initialize_projector_with_text()

        self.graph_output = 18
        self.spatial_graph = SpatialConceptGraph(
            num_concepts=6,
            pool_size=7
        ).to(device)

        # ==========================================
        # ★★★ 核心修改：混合特征维度 (Hybrid Dimensions) ★★★
        # ==========================================
        # 概念特征维度: 6个概念 * 3种池化(Mean, Max, Min) = 18维
        # 图特征压缩到 64 维 64 + 18 = 82
        concept_dim = self.num_concepts * 3
        fused_dim = concept_dim + self.graph_output

        self.headx = nn.Linear(concept_dim, 5)
        self.headx.to(device)

        # 4. Auxiliary Head (Lesion Classification) -> 依然只接收 18 维的概念特征 用于约束 projection 的训练
        self.aux_head = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_concepts)
        )
        self.aux_head.to(device)

        # 新增一个专门用于病灶预测的分类头，用于约束训练 graph 图网络层
        self.lesion_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_concepts) # 输出 6 个病灶的概率
        ).to(device)

        #
        # final_dim = fused_dim + 768
        final_dim = fused_dim
        self.final_headx = nn.Linear(final_dim, 5)
        self.final_headx.to(device)

        # self.fh = nn.Linear(768, 5)
        # self.fh.to(device)


    def _initialize_projector_with_text(self):
        """
        使用 CLIP 的 Text Encoder 生成的文本特征，来初始化 1x1 卷积的权重。
        这样卷积操作就等价于计算图像特征和文本特征的相似度。
        """
        print("Initializing concept projector with text embeddings...")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")

        prompts = [f"一张包含{c}的眼底照片" for c in self.concepts]
        text_inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            # 获取文本特征 [C, 768] (假设投影后是 512，这里需要注意维度匹配)
            # CLIP 的 visual output 是 512 维 (经过 proj)，但 transformer output 是 768 维
            # 我们需要拿到 text 的最终 embedding
            text_output = self.clip_model.encode_text(text_inputs['input_ids'])
            text_feats = text_output[0]
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

            # 这是一个关键点：ViT 的 patch token 输出通常是 768 维，
            # 但 CLIP 有一个 visual.proj 矩阵把 768 -> 512
            # 文本特征是 512 维。
            # 所以我们的 mapping layer 应该是：
            # Input(768) -> [Visual Proj (768x512)] -> [Text Feats (512xC)]^T -> Output(C)

            # 为了简化，我们直接把 Visual Proj 和 Text Feats 乘起来，合成一个矩阵
            # Weight = (Text_Feats @ Visual_Proj.T).T -> Shape [C, 768]

            visual_proj = self.visual.proj.to(self.device) # [768, 512]

            # 合成权重: [C, 512] x [512, 768] = [C, 768]
            # 注意转置关系
            fused_weight = text_feats @ visual_proj.t()

            # 赋值给卷积层
            # Conv2d weight shape: [Out, In, kH, kW] -> [C, 768, 1, 1]
            self.concept_projector.weight.data = fused_weight.view(self.num_concepts, 768, 1, 1)

            # 我们可以选择冻结它（模拟 Zero-shot），也可以放开微调（Training the bottleneck）
            # 论文中通常会放开微调一下
            self.concept_projector.weight.requires_grad = True

    def forward(self, x):
        # --- 1. Backbone 前向传播 ---
        with torch.no_grad():
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.visual.ln_post(x[:, 1:, :])
            B, L, D = x.shape
            H = W = int(L**0.5)
            # x shape: [B, 768, 14, 14]
            x = x.permute(0, 2, 1).reshape(B, D, H, W)

        # --- 2. 提取特征 ---
        # B. 提取概念特征 (Concept Maps & Pooling)
        concept_maps = self.concept_projector(x) # [B, 6, 14, 14]
        c_mean = concept_maps.mean(dim=(2, 3))   # [B, 6]
        c_max = concept_maps.amax(dim=(2, 3))    # [B, 6]
        c_min = concept_maps.amin(dim=(2, 3))    # [B, 6]

        global_features = x.mean(dim=(2, 3))

        # C. 拼接概念特征 -> [B, 18]
        concept_features = torch.cat([c_mean, c_max, c_min], dim=1)

        # 辅助分类头只接收概念特征 (18维)，保证 Stage 1 逻辑不受影响，用于约束 projection 训练
        lesion_logits_aux = self.aux_head(concept_features)

        # D. 获取图推理特征
        graph_features, after_graph_map = self.spatial_graph(concept_maps, lesion_logits_aux) # [B, 64]

        # F. 图推理特征联合初始概念池化特征
        fused_features = torch.cat([concept_features, graph_features], dim=1)

        # 主分级头接收融合特征 (82维)
        grade_logits = self.headx(concept_features)


        # grade_logits_final = self.fh(global_features)

        # 最终概念分类头接收概念特征（18维）和图推理特征（64维）约束图推理参数训练
        lesion_logits_graph = self.lesion_head(fused_features) # [B, 6]

        # 最终分类头接收概念特征（18维）和图推理特征（18维）
        global_features = x.mean(dim=(2, 3))
        final_concept = torch.cat([fused_features], dim=1)
        grade_logits_final = self.final_headx(final_concept)

        return grade_logits, concept_maps, lesion_logits_aux, lesion_logits_graph, grade_logits_final, after_graph_map

# === 测试代码 ===
if __name__ == "__main__":
    CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "软性渗出", "玻璃体积血", "玻璃体混浊"]
    PATH = "./checkpoints/finetuned_model/dr_grading_finetune/checkpoints/epoch_latest.pt"

    model = SALF_CBM(PATH, CONCEPTS)

    dummy_img = torch.randn(2, 3, 224, 224).cuda()
    logits, maps = model(dummy_img)

    print(f"Output Logits: {logits.shape}")   # [2, 5]
    print(f"Concept Maps: {maps.shape}")     # [2, 6, 14, 14]
    print("模型构建成功！这才是原论文的完整形态。")
