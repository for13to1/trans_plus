# NumPy Autograd Transformer (Tiny Trans Plus)

这个项目实现了一个完全基于 NumPy 的微型 Transformer 模型，不依赖 PyTorch 或 TensorFlow 等深度学习框架。为了实现反向传播和模型训练，核心包含了一个仅使用 NumPy 构建的简易 Autograd（自动微分）引擎。

目的是为了深入理解 Transformer 的内部工作原理以及自动微分机制的底层实现。

## 🎯 项目特点

*   **纯 NumPy 实现**: 所有的矩阵运算、前向传播、反向传播均只使用 NumPy。
*   **手写 Autograd**: 实现了一个具有 `backward()` 功能的 `Tensor` 类，支持计算图构建和梯度自动推导。
*   **完整 Transformer**: 包含 Multi-Head Attention, LayerNorm, FeedForward, Positional Encoding 等核心组件。
*   **由简入繁**: 以简单的“两位数加法”任务为例，演示模型的训练和推理过程。

## 📂 文件结构

```text
.
├── autograd.py   # 核心自动微分引擎：定义 Tensor 类及各种算子 (Add, Mul, MatMul, ReLU, Softmax 等) 的反向传播逻辑
├── model.py      # 模型定义：基于 autograd.py 构建的 Linear, LayerNorm, Attention, TransformerBlock 等层
├── dataset.py    # 数据集：生成随机的加法算式 (e.g., "12+34=46") 用于训练
├── train.py      # 训练脚本：构建模型、定义 Loss、优化器 (Adam) 并执行训练循环
├── inference.py  # 推理脚本：加载训练好的权重，进行交互式的加法预测
└── README.md     # 说明文档
```

## 🚀 快速开始

### 1. 安装依赖

仅需安装 `numpy`：

```bash
pip install numpy
```

### 2. 训练模型

运行 `train.py` 开始训练模型。代码会生成加法数据，并在终端打印 Loss 和部分预测结果。

```bash
python3 train.py
```

训练过程说明：
*   默认训练 3000 步。
*   模型参数非常小（微型配置）。
*   训练完成后，权重会保存为 `tiny_model.pkl`。

### 3. 模型推理

训练完成后，运行 `inference.py` 测试模型效果。你可以在终端输入加法算式，查看模型输出。

```bash
python3 inference.py
```

**示例交互：**

```text
=== Tiny Transformer Inference ===
Weights loaded from tiny_model.pkl
Type a 2-digit addition problem (e.g. '12+34') or 'q' to quit.

Problem > 23+45
Model   > 68
Check   > ✅

Problem > 99+01
Model   > 100
Check   > ✅
```

## 🧠 核心实现细节

### Autograd Engine (`autograd.py`)
*   **Tensor 类**: 封装了 `data` (numpy array) 和 `grad`。
*   **动态图**: 每次运算都会创建一个新的 Tensor，并记录操作类型 (`_op`) 和前驱节点 (`_children`)，形成 DAG（有向无环图）。
*   **Backward**: 通过拓扑排序（Topological Sort）遍历计算图，调用每个算子定义的 `_backward` 闭包函数计算梯度。

### Transformer (`model.py`)
*   **Embedding**: 简单的查表实现，梯度通过 `np.add.at` 回传。
*   **MultiHeadAttention**: 实现了 `split_heads` 和 scaled dot-product attention。注意这里手动处理了 `transpose` 和 `reshape` 的维度变换。
*   **LayerNorm**: 手写了 Layer Normalization 的前向和反向梯度计算。

## ⚠️ 注意事项
*   本项目仅供学习和教学使用，性能和数值稳定性无法与成熟框架（PyTorch/JAX）相比。
*   Autograd 引擎实现较为基础，仅支持本项目所需的算子。

---
Enjoy coding with NumPy! 🧮
