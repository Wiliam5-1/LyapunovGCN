# 图神经网络优化算法与低精度推理框架

## 📖 项目概述

这是一个基于图神经网络（GNN）的节点分类优化框架，专注于解决传统GNN在深层传播中的梯度消失、过平滑和信息衰减问题。项目实现了创新的激活函数设计、权重约束机制，并提供了完整的INT8量化推理支持。

## 🎯 核心特性

- **创新模型**: NewActivationGNN - 基于分段激活函数和列随机权重约束的新型GNN
- **完整基准测试**: 5个标准基线模型（GCN、GAT、APPNP、SGC、GCNII）对比
- **低精度推理**: 完全自定义的INT8量化引擎，支持纯INT8域计算
- **大规模实验**: 10折交叉验证，多GPU并行，网格搜索超参数优化
- **梯度分析**: 训练稳定性监控和信号增益分析

## 📁 项目结构

### 核心文件

```
├── grid_search_v6.py          # 🎯 主程序 - 网格搜索、训练、评估
├── models_v6.py               # 🧠 模型定义 - 创新模型和基线实现
├── datasets_v5.py             # 📊 数据加载 - 标准图数据集处理
├── full_int8_inference.py     # ⚡ INT8推理引擎 - 纯INT8域计算
├── inc_quantization.py        # 🔧 INT8接口包装 - 量化API封装
└── algorithm_overview.md      # 📋 算法详解 - 技术文档
```

### 辅助工具

```
├── run_four_baselines.py      # 🏃 基线模型批量测试
├── convergence_experiment.py  # 📈 收敛性实验
├── gcn_convergence_test.py    # 🧪 GCN收敛测试
├── plot_activation_comparison.py # 📊 激活函数对比可视化
├── TEST_ACTIVATION.py         # 🧪 激活函数测试
```

## 🧠 模型架构

### 1. NewActivationGNN (主要创新模型)

#### 架构特点
- **三段激活函数**: 分段响应信号强度，结合ReLU稀疏性和Sigmoid平滑性
- **列随机权重约束**: [0,1]均匀初始化，训练后投影确保每列和为1
- **自适应残差连接**: 残差权重从0线性增长到0.5，防止深层过平滑
- **训练时归一化**: 只在预处理时归一化一次，后续保留原始特征值

#### 前向传播流程
```python
# 1. 初始变换
H⁰ = ReLU(X × W_in)

# 2. H0预处理（gamma参数控制强度）
H₀ = (1-γ) × H⁰ + γ × 1 × ReLU(c)ᵀ
H₀ = normalize(H₀)  # 归一化到[0,1]

# 3. 逐层更新 (L层)
for l in range(L):
    # 图卷积传播
    z = D⁻¹A × X_l × W_l

    # 三段激活函数
    X_{l+1} = δ_ε(z)

    # 自适应残差连接 (beta随层数增加)
    β_l = min(0.5, (l+1)/L × 0.5)
    X_{l+1} = (1-β_l) × X_{l+1} + β_l × H₀

    # Dropout（仅训练时）
    X_{l+1} = Dropout(X_{l+1})

# 4. 输出层
ŷ = log_softmax(X_L × W_out)
```

#### 三段激活函数 δ_ε(z)

定义：
```
δ_ε(x) = {
    0,                                      if x ∈ [0, x₁]
    θ × (1 + e^(-c)) / (1 + e^(-cθ)),      if x ∈ (x₁, x₂]
    1,                                      if x ∈ (x₂, 1]
}
其中 θ = (x - x₁) / (x₂ - x₁)
```

- **第一段** [0, x₁]: 输出0（抑制弱信号和噪声）
- **第二段** (x₁, x₂]: sigmoid平滑过渡（x₁=0.2, x₂=0.9）
- **第三段** (x₂, 1]: 输出1（完全保留强信号）

### 2. 基线模型对比

项目实现了5个标准GNN基线进行全面对比：

| 模型 | 年份 | 特点 | 激活函数 |
|------|------|------|----------|
| **GCN** | 2017 | 图卷积基础 | ReLU |
| **GAT** | 2018 | 注意力机制 | ELU |
| **APPNP** | 2019 | 个性化传播 | ReLU |
| **SGC** | 2019 | 简化卷积 | 无激活 |
| **GCNII** | 2020 | 初始残差 | ReLU |

## ⚡ INT8量化推理系统

### 架构设计

项目实现了完全自定义的INT8推理引擎，支持纯INT8域计算：

```
量化流程: FP32 → INT8 → 计算 → 反量化 → FP32输出
          ↓       ↓       ↓       ↓       ↓
        权重     激活     激活函   矩阵乘   接口兼容
```

### 核心组件

#### 1. INT8ActivationFunction - INT8域激活函数
```python
class INT8ActivationFunction:
    """基于查找表的INT8激活函数"""
    def __init__(self, activation_type='three_segment', ...):
        # 创建256个INT8值的查找表（-128到127）
        self.lut = self._create_lookup_table()

    def apply(self, x_int8):
        # O(1)查找表查询
        indices = (x_int8 + 128).long()
        return self.lut[indices]
```

#### 2. INT8MatMul - INT8矩阵乘法
```python
class INT8MatMul:
    @staticmethod
    def matmul(a_int8, a_scale, b_int8, b_scale):
        # 转换为INT32类型（准备计算）
        a_int32 = a_int8.to(torch.int32)
        b_int32 = b_int8.to(torch.int32)

        # 矩阵乘法：INT8×INT8元素相乘 + INT32累加
        # C[i,j] = Σ A[i,k] × B[k,j]
        # - 元素相乘: INT8 × INT8 → 结果范围[-16384, 16129]
        # - 累加求和: 使用INT32累加器避免溢出
        c_int32 = torch.matmul(a_int32, b_int32)

        # 反量化到FP32（缩放）
        c_fp32 = c_int32.float() * a_scale * b_scale

        # 重新量化回INT8
        c_int8 = torch.clamp(torch.round(c_fp32 / output_scale), -128, 127)
```

#### 3. FullINT8Model - 完整INT8推理包装器
```python
class FullINT8Model(nn.Module):
    def __init__(self, original_model, example_inputs, activation_params):
        # 1. 量化所有权重
        self.weight_int8_dict = {}
        self.weight_scale_dict = {}
        self._quantize_all_weights()

        # 2. 创建INT8激活函数
        self.int8_activation = INT8ActivationFunction(**activation_params)

        # 3. 替换PyTorch激活函数为INT8版本
        self._replace_activations_with_int8()
```

### 量化策略

- **权重量化**: 对称量化 `scale = max(|W|) / 127`
- **激活量化**: 动态量化，根据当前激活值范围计算scale
- **混合精度计算**: INT8元素相乘 → INT32累加 → FP32缩放 → INT8输出
- **查找表激活**: 预计算所有INT8输入的激活函数输出（256个条目）

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric networkx scipy matplotlib scikit-learn
```

### 基本用法

#### 1. 网格搜索优化模型超参数
```bash
# 在Texas数据集上搜索最佳gamma值
python grid_search_v6.py --dataset texas --gamma_values 0.2

# 多GPU并行搜索
python grid_search_v6.py --dataset texas --gpus 0,1,2 --gamma_values 0.1,0.2,0.3
```

#### 2. 运行基准测试
```bash
# 测试所有基线模型
python grid_search_v6.py --run_baselines --dataset texas

# 启用INT8推理加速
python grid_search_v6.py --run_baselines --dataset texas --use_int8_inference
```

#### 3. 时间性能对比
```bash
# 跨数据集时间基准测试
python grid_search_v6.py --benchmark_time --datasets texas,cora,chameleon
```

### 命令行参数

| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| `--dataset` | str | 数据集名称 | 'texas' |
| `--gamma_values` | float | 预处理强度参数 | [0.2] |
| `--c_activation` | float | 激活函数常数 | -0.5 |
| `--epsilon` | float | 激活函数阈值 | 0.1 |
| `--gpus` | str | GPU设备ID | '0' |
| `--run_baselines` | flag | 运行基线模型对比 | False |
| `--use_int8_inference` | flag | 启用INT8推理 | False |
| `--benchmark_time` | flag | 执行时间基准测试 | False |

## 📊 数据集支持

支持标准图神经网络基准数据集：

| 数据集 | 节点数 | 边数 | 特征维度 | 类别数 | 备注 |
|--------|--------|------|----------|--------|------|
| **Texas** | 183 | 325 | 1,703 | 5 | 网页分类 |
| **Cora** | 2,708 | 5,429 | 1,433 | 7 | 论文引用 |
| **Citeseer** | 3,327 | 4,732 | 3,703 | 6 | 论文引用 |
| **Pubmed** | 19,717 | 44,338 | 500 | 3 | 论文引用 |
| **Chameleon** | 2,277 | 36,101 | 2,325 | 5 | 网页图 |
| **Squirrel** | 5,201 | 217,073 | 2,089 | 5 | 网页图 |

## 🏗️ 核心算法详解

### 1. 三段激活函数设计

项目创新性地设计了三段激活函数，结合ReLU的稀疏性和Sigmoid的平滑性：

```python
def three_segment_activation(x, x1, x2, c):
    """
    三段激活函数 δ_ε(x)
    
    δ_ε(x) = {
        0,                                if x ∈ [0, x1]
        θ * (1 + e^(-c)) / (1 + e^(-cθ)), if x ∈ (x1, x2]
        1,                                if x ∈ (x2, 1]
    }
    
    其中 θ = (x - x1) / (x2 - x1)
    """
    mask1 = (x <= x1).float()  # [0, x1]
    mask2 = ((x > x1) & (x <= x2)).float()  # (x1, x2]
    mask3 = (x > x2).float()  # (x2, 1]
    
    # 第二段：sigmoid插值
    theta = (x - x1) / (x2 - x1)
    numerator = theta * (1 + torch.exp(torch.tensor(-c)))
    denominator = 1 + torch.exp(-c * theta)
    segment2 = numerator / denominator
    
    # 组合三段
    return mask1 * 0 + mask2 * segment2 + mask3 * 1
```

**参数设置**：
- x₁ = 0.2（第一段结束点）
- x₂ = 0.9（第二段结束点）
- c = -0.484754509027（sigmoid参数，控制平滑度）

### 2. 列随机权重约束

通过阈值过滤、最大值限制和归一化确保权重矩阵的列随机性：

```python
def project_column_stochastic(weight, max_weight=0.05, threshold=0.0):
    """列随机权重投影"""
    with torch.no_grad():
        # 1. 阈值过滤（抑制过小权重）
        weight.clamp_(min=threshold)
        weight[weight < threshold] = 0

        # 2. 最大值限制（防止权重坍缩）
        weight.clamp_(max=max_weight)

        # 3. 列归一化（确保每列和为1）
        col_sums = weight.sum(dim=0, keepdim=True)
        col_sums = torch.clamp(col_sums, min=1e-8)
        weight.div_(col_sums)
```

**权重初始化**：
```python
# 在[0,1]之间均匀采样
weight.data.uniform_(0, 1)
# 然后进行列随机投影
project_column_stochastic(weight)
```

### 3. 自适应残差连接

根据网络深度动态调整残差强度：

```python
# 残差权重随层数线性增长：从0到0.5
beta_l = min(0.5, (layer_idx + 1) / total_layers * 0.5)

# 更新公式
X_{l+1} = δ_ε(D^{-1}AX_lW_l)  # 三段激活函数
X_{l+1} = (1 - beta_l) * X_{l+1} + beta_l * H_initial

# 浅层（β≈0）：主要学习新特征，避免过早平滑
# 深层（β≈0.5）：50%保留初始特征，防止过度平滑
```

## 📈 梯度分析与训练稳定性

### 1. 梯度方差监控

```python
# 计算学习率归一化的梯度方差
normalized_grad_variance = mean(layer_grad_vars) / (learning_rate ** 2)

# 反映权重更新的稳定性
# 较小的方差表示更稳定的梯度流
```

### 2. 信号增益分析

```python
# 计算权重矩阵的信号放大能力
for layer in range(num_layers):
    weight_product = product of all subsequent weight matrices
    signal_gain = max(column_sums(abs(weight_product)))
```

## 🔧 技术实现亮点

### 1. 多GPU并行执行

```python
def run_single_experiment(exp_idx, gamma_val, alpha_val, lambda_val, c_val, gpu_id):
    """每个GPU独立运行一个实验配置"""
    device = torch.device(f'cuda:{gpu_id}')

    # 模型初始化和训练
    model = create_model(...)
    result = k_fold_cv(model, ...)

    return result
```

### 2. 内存优化策略

- GPU内存按需清理：`torch.cuda.empty_cache()`
- 数据分批拷贝：避免多个GPU同时保留完整数据集
- 显存同步：`torch.cuda.synchronize()`

### 3. 10折分层交叉验证

```python
# 按类别分层划分，确保每折类别分布一致
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_val_idx, test_idx) in enumerate(skfold.split(X, y)):
    # 在训练+验证集中进一步划分
    train_idx, val_idx = train_test_split(train_val_indices,
                                         test_size=1/9, stratify=labels)
```

## 🎯 使用示例

### 1. 超参数搜索

```bash
# 单GPU搜索gamma参数
python grid_search_v6.py --dataset texas --gamma_values 0.1,0.2,0.3,0.4,0.5

# 多GPU并行搜索（每个GPU运行一个配置）
python grid_search_v6.py --dataset cora --gpus 0,1,2,3 --gamma_values 0.2 --c_activation -0.5,-0.7,-0.9
```

### 2. 基线模型对比

```bash
# 在所有数据集上测试所有基线模型
python grid_search_v6.py --run_baselines --datasets texas,cora,chameleon,citeseer,squirrel,pubmed

# 只测试GCN和GAT
python grid_search_v6.py --run_baselines GCN,GAT --dataset texas
```

### 3. INT8推理加速

```bash
# 启用INT8推理进行基准测试
python grid_search_v6.py --run_baselines --dataset texas --use_int8_inference

# 验证INT8推理精度
python grid_search_v6.py --dataset texas --gamma_values 0.2 --use_int8_inference --compare_training
```

### 4. 时间性能分析

```bash
# 跨数据集时间基准测试
python grid_search_v6.py --benchmark_time --datasets texas,cora --gpus 0,1

# 生成性能对比表格
python grid_search_v6.py --benchmark_time --output_format table
```

## 📋 文件说明

### 核心模块

- **`grid_search_v6.py`**: 主程序入口，支持网格搜索、基准测试、时间分析等多种模式
- **`models_v6.py`**: 模型定义，包含NewActivationGNN和5个基线模型的实现
- **`datasets_v5.py`**: 数据加载器，支持标准图数据集的预处理和加载
- **`full_int8_inference.py`**: INT8推理引擎，实现纯INT8域的矩阵运算和激活函数
- **`inc_quantization.py`**: INT8量化接口，提供易用的量化API

### 实验工具

- **`run_four_baselines.py`**: 批量运行基线模型的便捷脚本
- **`convergence_experiment.py`**: 收敛性实验分析
- **`plot_activation_comparison.py`**: 激活函数性能可视化

### 文档和结果

- **`algorithm_overview.md`**: 详细的算法技术文档
- **`results/`**: 实验结果和日志文件
- **`data/`**: 原始数据集存储目录
