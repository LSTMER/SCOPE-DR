# SCOPE-DR

SCOPE-DR 是一个面向糖尿病视网膜病变（Diabetic Retinopathy, DR）分级的研究型项目，基于 `RET_CLIP` 视网膜基础模型构建，并在其上融合了概念瓶颈学习、空间概念图推理、MIL-VT 分支和渐进式消融实验流程。

这个仓库的核心目标不是单纯复现上游 `RET_CLIP`，而是围绕 DR 分级任务构建完整的实验链路，包括：

- 基于眼底图像的 5 分类 DR 分级
- 病灶概念监督与概念矩阵蒸馏
- 空间概念图建模
- MIL-VT 与概念图的融合推理
- `vit_direct / cp / cp_graph / full` 四种渐进式消融实验

## 项目特点

- 使用 `RET_CLIP` 作为视觉-语言基础骨干
- 支持将 CSV + 图像目录打包为 LMDB
- 支持为数据集批量生成概念矩阵
- 支持概念瓶颈模型与图推理模型训练
- 支持 MIL-VT 分支训练与融合
- 支持完整消融训练、评估和结果汇总
- 提供病例分析和分级分布可视化脚本

## 目录结构

```text
SCOPE-DR-main/
|- RET_CLIP/                        # RET-CLIP 主体代码与上游训练/评估模块
|- MultiModalDataset*.py           # 数据集读取逻辑（训练 / 验证 / 消融）
|- graph_model_cbm*.py             # 概念瓶颈与融合模型定义
|- DynamicGraphProportion.py       # 空间概念图模块
|- mil_vt_model.py                 # MIL-VT 分支模型
|- toLMDB.py                       # 将 CSV + 图像目录转换为 LMDB
|- generate_dataset_matrix.py      # 为整个数据集生成概念矩阵
|- train_mil_vt.py                 # 训练 MIL-VT 分支
|- train_salf_cbm_end2end.py       # SALF-CBM 训练流程
|- train_fusion_cbm.py             # 融合模型训练流程
|- train_fusion_cbm_ablation.py    # 渐进式消融训练
|- evaluate_cbm.py                 # 标准评估
|- evaluate_cbm_ablation.py        # 消融评估
|- run_ablation_and_summarize.py   # 汇总多阶段消融结果
|- run_full_ablation_pipeline.py   # 一键跑完整消融流程
|- visualize_case_analysis_56.py   # 病例可视化分析
|- visualize_grade_distribution.py # 分级分布可视化
|- run_finetune.sh                 # RET-CLIP 微调示例脚本
`- requirements.txt                # 基础依赖
```

## 环境依赖

建议环境：

- Python >= 3.8
- PyTorch
- torchvision
- CUDA 环境（如需 GPU 训练）

安装基础依赖：

```bash
pip install -r requirements.txt
```

注意：当前仓库中的部分可视化和分析脚本还依赖一些没有完整写入 `requirements.txt` 的包，常见包括：

- `matplotlib`
- `Pillow`
- `opencv-python`
- `seaborn`

## 数据准备

当前代码保留了较多研究阶段写法，不少脚本中仍然直接写死了本地绝对路径。在正式运行前，请先根据你的环境修改相应脚本中的数据路径和权重路径。

推荐的数据准备顺序如下：

1. 准备标注 CSV 和眼底图像目录
2. 使用 `toLMDB.py` 将数据集转换为 LMDB
3. 使用 `generate_dataset_matrix.py` 生成概念矩阵
4. 准备 `RET_CLIP` 预训练权重

## 推荐实验流程

### 1. 构建 LMDB

```bash
python toLMDB.py
```

### 2. 生成概念矩阵

```bash
python generate_dataset_matrix.py
```

### 3. 训练 MIL-VT 分支

```bash
python train_mil_vt.py
```

### 4. 运行消融训练

单阶段训练示例：

```bash
python train_fusion_cbm_ablation.py --ablation-stage full --run-stage1 --run-stage4
```

完整渐进式流程示例：

```bash
python run_full_ablation_pipeline.py --stages vit_direct cp cp_graph full
```

### 5. 评估模型

```bash
python evaluate_cbm_ablation.py --ablation-stage full --checkpoint path/to/stage4_final.pth
```

### 6. 汇总实验结果

```bash
python run_ablation_and_summarize.py --output-dir evaluation_results/ablation_summary
```

## 关键说明

- 这个仓库顶层已经不只是上游 `RET_CLIP`，而是围绕 DR 分级做过二次开发的研究代码集合。
- 多个训练脚本之间存在阶段性依赖，后续阶段通常需要前一阶段的 checkpoint。
- `MultiModalDataset.py`、`MultiModalDataset1.py`、`MultiModalDataset2.py` 分别服务于不同训练/评估流程，不建议在不了解调用关系的前提下直接合并。
- 许多配置仍保留研究环境中的绝对路径，迁移到新机器前需要先统一修改。

## 当前保留的核心脚本

- 数据构建：`toLMDB.py`、`generate_dataset_matrix.py`
- 骨干与上游代码：`RET_CLIP/`
- 核心模型：`graph_model_cbm.py`、`graph_model_cbm_fusion_v2.py`、`graph_model_cbm_fusion_v2_ablation.py`
- 训练流程：`train_salf_cbm_end2end.py`、`train_fusion_cbm.py`、`train_fusion_cbm_ablation.py`、`train_mil_vt.py`
- 评估流程：`evaluate_cbm.py`、`evaluate_cbm_ablation.py`、`evaluate_fusion.py`
- 实验编排：`run_full_ablation_pipeline.py`、`run_ablation_and_summarize.py`

## 清理说明

为了让仓库更适合整理到 GitHub，我已经移除了几类明显不属于主流程、且没有被主训练/评估链路引用的孤立脚本，包括：

- `11.py`
- `dataSet.py`
- `demo.py`
- `eva.py`
- `eva1.py`
- `fine-tuned.py`
- `test.py`
- `test_dataset.py`
- `test_stu.py`

这些文件主要属于草稿、旧测试、临时演示或重复实验脚本，删除后不会影响当前保留的主流程代码。

## 致谢

本项目基于 `RET_CLIP` 相关工作扩展而来。如用于论文、实验报告或公开项目，建议同时引用上游 `RET_CLIP` 及其对应论文与资源。
