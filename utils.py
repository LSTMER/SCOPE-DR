import torch
import torch.nn as nn
import loralib as lora
from RET_CLIP.clip.utils import create_model

def load_ret_clip_with_lora(
    base_model_path: str,
    lora_weight_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    r: int = 8,
    lora_alpha: int = 16,
    verbose: bool = True
) -> nn.Module:
    """
    加载基础 RET-CLIP 模型并注入双塔 LoRA 权重。

    Args:
        base_model_path: 基础未微调模型的路径 (例如 ret-clip.pt)
        lora_weight_path: 微调后保存的 LoRA 权重路径 (例如 epoch2.pt)
        device: 'cuda' 或 'cpu'
        r: LoRA 的秩 (需与训练时保持一致)
        lora_alpha: LoRA 的缩放系数 (需与训练时保持一致)
        verbose: 是否打印详细的加载报告

    Returns:
        注入了新知识并处于 eval() 状态的 CLIP 模型
    """
    if verbose: print(f"==> 1. 加载基础模型: {base_model_path}")
    raw_ckpt = torch.load(base_model_path, map_location="cpu")

    # 检查并修正权重结构
    if isinstance(raw_ckpt, dict):
        if "state_dict" in raw_ckpt:
            clean_checkpoint = raw_ckpt
        elif "model" in raw_ckpt:
            clean_checkpoint = {"state_dict": raw_ckpt["model"]}
        else:
            clean_checkpoint = {"state_dict": raw_ckpt}
    else:
        raise ValueError("错误: 基础权重文件格式不对 (不是字典)")

    # 创建基础模型
    model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=clean_checkpoint)

    if verbose: print("==> 2. 注入空的双塔 LoRA 旁路...")
    def replace_linear_with_lora(module, target_keywords):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(kw in name for kw in target_keywords):
                lora_layer = lora.Linear(
                    child.in_features, child.out_features,
                    r=r, lora_alpha=lora_alpha, bias=(child.bias is not None)
                )
                lora_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    lora_layer.bias.data = child.bias.data.clone()
                setattr(module, name, lora_layer)
            else:
                replace_linear_with_lora(child, target_keywords)

    # 视觉 + 文本双塔注入
    target_layers = [
        "q_proj", "v_proj", "c_fc", "c_proj", "out_proj", # 视觉
        "query", "value", "dense"                         # 文本
    ]
    replace_linear_with_lora(model, target_layers)

    if verbose: print(f"==> 3. 加载微调 LoRA 权重: {lora_weight_path}")
    lora_ckpt = torch.load(lora_weight_path, map_location='cpu')

    # 智能清洗 DDP 前缀
    state_dict_to_use = lora_ckpt.get('state_dict', lora_ckpt)

    if verbose:
        print(f"    📦 Checkpoint 文件内含权重数量: {len(state_dict_to_use)}")

    cleaned_lora_ckpt = {}
    for k, v in state_dict_to_use.items():
        # 暴力清洗一切多卡/包装器前缀
        new_key = k.replace("module.clip_model.", "").replace("module.model.", "").replace("module.", "")
        cleaned_lora_ckpt[new_key] = v

    # 加载权重，并捕获结果
    load_result = model.load_state_dict(cleaned_lora_ckpt, strict=False)

    if verbose:
        lora_keys_in_dict = [k for k in cleaned_lora_ckpt.keys() if 'lora' in k]
        unexpected_lora_keys = [k for k in load_result.unexpected_keys if 'lora' in k]
        print(f"    🔍 尝试加载的 LoRA 参数量: {len(lora_keys_in_dict)}")
        print(f"    ✅ 成功加载的 LoRA 参数量: {len(lora_keys_in_dict) - len(unexpected_lora_keys)}")
        if len(unexpected_lora_keys) > 0:
            print(f"    ❌ 警告: 有 {len(unexpected_lora_keys)} 个权重找不到坑位！示例: {unexpected_lora_keys[:2]}")

    # ==========================================
    # 4. 彻底冻结所有参数并转为推理模式
    # ==========================================
    for param in model.parameters():
        param.requires_grad = False

    model = model.float()
    model.to(device)
    model.eval()

    if verbose: print("==> 🎉 模型组装完毕，已切换至 eval 模式并挂载到目标设备！\n")

    return model
