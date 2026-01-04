# Trans Plus

这个项目实现了一个完全基于 NumPy 的微型 Transformer 模型，不依赖 PyTorch 或 TensorFlow 等深度学习框架。
为了实现反向传播和模型训练，核心包含了一个仅使用 NumPy 构建的简易 Autograd（自动微分）引擎。

目的是为了深入理解 Transformer 的内部工作原理以及自动微分机制的底层实现。

## 🎯 项目特点

- **纯 NumPy 实现**: 所有的矩阵运算、前向传播、反向传播均只使用 NumPy。
- **手写 Autograd**: 实现了一个具有 `backward()` 功能的 `Tensor` 类，支持计算图构建和梯度自动推导。
- **完整 Transformer**: 包含 Multi-Head Attention, LayerNorm, FeedForward, Positional Encoding 等核心组件。
- **灵活的算术任务**: 支持可配置位数的加法训练（目前默认为 2 位数加法），演示模型的序列建模能力。
- **集中配置**: 通过 `config.py` 管理模型超参数和训练设置。
- **自带分析工具**: 内置模型参数统计与显存估算工具 (`framework.analyze`)，自动分析模型规模。

## 📂 文件结构

```text
.
├── framework/           # Infrastructure
│   ├── __init__.py
│   ├── autograd.py      # Autograd Engine (Minimal)
│   ├── model.py         # Transformer Layers (Paper compliant)
│   └── analyze.py       # Parameter & Memory Analysis
├── addition_agent/      # Addition Task
│   ├── __init__.py
│   ├── config.py        # Task Configuration
│   ├── dataset.py       # Addition Data Generator
│   ├── train.py         # Training Script
│   └── inference.py     # Inference Script
│
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

仅需安装 `numpy`：

```bash
pip install numpy
```

### 2. 训练模型

运行 `addition_agent/train.py` 开始训练模型。

```bash
python3 addition_agent/train.py
```

训练过程说明：
- 默认训练 3000 步。
- 配置参数可在 `addition_agent/config.py` 中修改（如 `MAX_DIGITS`, `BATCH_SIZE` 等）。
- 训练完成后，权重会保存为 `addition_model.pkl`。

### 3. 模型推理

训练完成后，运行 `addition_agent/inference.py` 测试模型效果。

```bash
python3 addition_agent/inference.py
```

**示例交互：**

```text
=== Tiny Transformer Inference ===
Weights loaded from addition_model.pkl
Type a addition problem (e.g. '12+34') or 'q' to quit.

Problem > 12+34
Model   > 46
Check   > ✅

Problem > 23+45
Model   > 68
Check   > ✅
```

## 🧠 核心实现细节

### Autograd Engine (`autograd.py`)

- **Tensor 类**: 封装了 `data` (numpy array) 和 `grad`。
- **动态图**: 每次运算都会创建一个新的 Tensor，并记录操作类型 (`_op`) 和前驱节点 (`_children`)，形成 DAG（有向无环图）。
- **Backward**: 通过拓扑排序（Topological Sort）遍历计算图，调用每个算子定义的 `_backward` 闭包函数计算梯度。

### Transformer (`model.py`)

- **Embedding**: 简单的查表实现，梯度通过 `np.add.at` 回传。
- **MultiHeadAttention**: 实现了 `split_heads` 和 scaled dot-product attention。注意这里手动处理了 `transpose` 和 `reshape` 的维度变换。
- **LayerNorm**: 手写了 Layer Normalization 的前向和反向梯度计算。
- **Positional Encoding**: 支持动态长度的位置编码，适应不同位数的加法任务。

## ⚠️ 注意事项

- 本项目仅供学习和教学使用，性能和数值稳定性无法与成熟框架（PyTorch/JAX）相比。
- Autograd 引擎实现较为基础，仅支持本项目所需的算子。
