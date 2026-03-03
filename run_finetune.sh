#!/usr/bin/env bash

# === 1. 显卡与环境设置 ===
export CUDA_VISIBLE_DEVICES=6  # 指定使用第0号显卡
echo "DEBUG: Setting CUDA_VISIBLE_DEVICES to $CUDA_VISIBLE_DEVICES"
# 单卡训练配置
GPUS_PER_NODE=1
WORKER_CNT=1
export MASTER_ADDR=localhost
export MASTER_PORT=8511
export RANK=0

# 把当前目录加入 Python 路径，防止找不到模块
export PYTHONPATH=${PYTHONPATH}:`pwd`/RET_CLIP/

# === 2. 关键路径配置 (★★★ 请修改这里 ★★★) ===

# A. 你的 LMDB 数据文件夹路径 (刚才生成的那个文件夹)
# 如果你分了 train 和 val，请分别指定；如果没有 val，就都指向同一个
train_data="./lmdb_output/train_lmdb_latest"
val_data="./lmdb_output/train_lmdb_latest"

# B. 预训练权重路径 (原始的 RET-CLIP 权重)
resume="./RET_CLIP/checkpoint/ret-clip.pt"

# C. 训练结果保存路径
output_base_dir="./checkpoints/finetuned_model_x"
name="dr_grading_finetune"

# === 3. 微调参数设置 (已为你调优) ===

# 学习率 (LR): 微调必须用小学习率，防止破坏预训练知识
# 建议: 1e-5 到 5e-6
lr=1e-5

# Batch Size: 显存允许的情况下越大越好
# 3090/4090 (24G) 可以尝试 32 或 64
# 2080Ti/3080 (10G/12G) 建议 16 或 32
batch_size=32
valid_batch_size=32

# 训练轮数: 微调通常很快收敛，5-10 轮即可
max_epochs=10

# 文本长度: 你的描述不算太长，64 或 100 足够
context_length=77

# 模型架构 (保持不变)
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese

# === 4. 标志位设置 ===
# reset-optimizer: 因为是新任务，建议重置优化器状态，从头计算动量
reset_optimizer="--reset-optimizer"
# reset-data-offset: 数据从头开始读
reset_data_offset="--reset-data-offset"
# use-augment: 开启数据增强 (随机裁剪、翻转)，这对小数据集非常重要！
use_augment="--use-augment"

# === 5. 启动训练 ===
echo "开始微调训练..."
echo "加载权重: ${resume}"
echo "数据路径: ${train_data}"

CUDA_VISIBLE_DEVICES=6 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --use_env RET_CLIP/training/main.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_optimizer} \
          ${reset_data_offset} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-epoch-frequency=1 \
          --log-interval=10 \
          --report-training-batch-acc \
          --context-length=${context_length} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --lr=${lr} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          --text-model=${text_model} \
          ${use_augment} \
          --skip-aggregate
          # --grad-checkpointing \

