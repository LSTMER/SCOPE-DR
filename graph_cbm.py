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

        # ==========================================
        # ★★★ [新增] 引入符号化空间图推理模块 ★★★
        # ==========================================
        self.graph_out_dim = 16  # 图网络输出的每个节点的特征维度 (可调超参)
        self.graph_reasoning = SpatialConceptGraph(
            num_nodes=self.num_concepts,
            spatial_size=14,
            out_features=self.graph_out_dim,
            sigma=2.0,
            lambda_prior=0.1
        ).to(self.device)

        # 概念特征的总维度 = 节点数(6) * 每个节点的维度(128) = 768维
        fused_dim = self.num_concepts * self.graph_out_dim

        # ==========================================
        # ★★★ [修改] 更新分类头 ★★★
        # ==========================================
        # 4. 主分级头 (DR Grading) -> 接收 768 维的图融合特征
        # ViT-B 默认的全局特征维度是 768
        global_dim = 768
        self.graph_out_dim = 16
        graph_fused_dim = self.num_concepts * self.graph_out_dim
        # 拼接后的最终维度: 768 (Graph) + 768 (Global) = 1536
        self.head = nn.Sequential(
            # 第一道防线：强行把 18维统计和 6维Logits 拉到同一个起跑线上！(均值为0，方差为1)
            nn.BatchNorm1d(24),

            # 第二道防线：升维做非线性组合，寻找特征间的交叉关系
            nn.Linear(24, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3), # 防止这 24 维精华特征过拟合

            # 第三道防线：输出 5 分类
            nn.Linear(64, 5)
        ).to(self.device)

        # 5. 辅助分类头 (Lesion Classification)
        # [逻辑优化]：现在的图网络让每个节点(128维)专门负责一种病灶。
        # 因此，辅助头只需要一个能把 128维 -> 1维(置信度) 的小网络，应用到所有节点上。
        self.aux_head = nn.Sequential(
            nn.Linear(19, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

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
        concept_maps = self.concept_projector(x)
        node_embeddings, adj_matrix = self.graph_reasoning(concept_maps)

        # ==========================================
        # 🌊 第一流：经典统计特征 (保底，绝对不会崩)
        # ==========================================
        c_mean = concept_maps.mean(dim=(2, 3))   # [B, 6]
        c_max = concept_maps.amax(dim=(2, 3))    # [B, 6]
        c_min = concept_maps.amin(dim=(2, 3))    # [B, 6]
        stat_features = torch.cat([c_mean, c_max, c_min], dim=1) # [B, 18]
        stat_featuresa = torch.stack([c_mean, c_max, c_min], dim=2)

        # ==========================================
        # 🌊 第二流：图推理高阶关系 (拔高，抓遮挡和共现)
        # ==========================================
        # ==========================================
        # 🤝 融合：双流拼接 (18 + 96 = 114 维)
        # ==========================================
        # （记得把主分类头 self.head 的输入维度改成 114）
        # final_features = torch.cat([stat_features, graph_features], dim=1)
        aux_features = torch.cat([node_embeddings, stat_featuresa], dim = 2)

        # 3. 分类头预测

        lesion_logits = self.aux_head(aux_features).squeeze(-1) # 依然用节点特征预测病灶

        final_features = torch.cat([stat_features, lesion_logits], dim=1) # [B, 24]
        grade_logits = self.head(final_features)

        return grade_logits, concept_maps, lesion_logits, adj_matrix

# === 测试代码 ===
# === 测试代码 ===
if __name__ == "__main__":
    CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "软性渗出", "玻璃体积血", "玻璃体混浊"]

    # 随便写个假路径测试网络结构
    PATH = "dummy.pt"

    try:
        model = SALF_CBM(PATH, CONCEPTS)
    except Exception as e:
        print("注意: 真实运行需要有效权重，这里仅供代码逻辑检查。")
        # 强制跑通结构的 Mock
        class MockModel(nn.Module): # <--- 改为直接继承 nn.Module
            def __init__(self):
                super().__init__()  # <--- 正确的 super 调用
                self.device = 'cuda'
                self.num_concepts = 6
                self.concept_projector = nn.Conv2d(768, 6, 1, bias=False).cuda()
                self.graph_out_dim = 128
                self.graph_reasoning = SignedGraphReasoning(num_nodes=6, spatial_size=14, out_features=128).cuda()
                self.head = nn.Linear(6 * 128, 5).cuda()
                self.aux_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()

            def forward(self, x):
                # 模拟 backbone 已经提取出 768维 的特征图
                fake_x = torch.randn(x.size(0), 768, 14, 14, device=x.device)
                concept_maps = self.concept_projector(fake_x)
                node_embeddings, adj_matrix = self.graph_reasoning(concept_maps)
                graph_global_features = node_embeddings.view(x.size(0), -1)
                grade_logits = self.head(graph_global_features)
                lesion_logits = self.aux_head(node_embeddings).squeeze(-1)
                return grade_logits, concept_maps, lesion_logits, adj_matrix

        model = MockModel()

    # 造一个假的输入图片 (BatchSize=2, Channels=3, 224x224)
    dummy_img = torch.randn(2, 3, 224, 224).cuda()

    grade_logits, concept_maps, lesion_logits, adj_matrix = model(dummy_img)

    print(f"最终分级 Logits: \t {grade_logits.shape} \t\t (期望: [2, 5])")
    print(f"病灶预测 Logits: \t {lesion_logits.shape} \t\t (期望: [2, 6])")
    print(f"原始概念图 Maps: \t {concept_maps.shape} \t (期望: [2, 6, 14, 14])")
    print(f"动态邻接矩阵 Adj: \t {adj_matrix.shape} \t\t (期望: [2, 6, 6])")
    print("\n🚀 图推理融合成功！各层 Tensor 维度对齐完毕！")
