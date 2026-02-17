"""
Version 4.0 Models - 训练时归一化，推理时不归一化
基于Version 3.0，修改：X只在训练时归一化，推理时不归一化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================== FP8 Support Functions ========================

def convert_tensor_to_fp8(tensor, use_e4m3=True):
    """
    将张量转换为FP8格式
    
    Args:
        tensor: 输入张量
        use_e4m3: 如果为True使用float8_e4m3fn（适合前向传播），否则使用float8_e5m2（适合梯度）
    
    Returns:
        FP8格式的张量
    """
    if use_e4m3:
        # E4M3格式：4位指数，3位尾数，更大的数值范围，适合激活值和权重
        return tensor.to(dtype=torch.float8_e4m3fn)
    else:
        # E5M2格式：5位指数，2位尾数，动态范围更大，适合梯度
        return tensor.to(dtype=torch.float8_e5m2)


def convert_model_to_fp8(model, use_e4m3=True):
    """
    将模型的所有参数转换为FP8格式（用于推理）
    
    Args:
        model: PyTorch模型
        use_e4m3: 是否使用E4M3格式（默认True）
    
    注意：
        - 所有参数都会转换为FP8
        - 这是不可逆的操作，建议在推理前保存原始模型
        - 某些操作可能需要临时转换回FP16/FP32
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            try:
                param.data = convert_tensor_to_fp8(param.data, use_e4m3)
            except Exception as e:
                # 如果转换失败（某些参数类型不支持FP8），保持原始精度
                print(f"Warning: Failed to convert {name} to FP8: {e}. Keeping original dtype.")
    
    # 标记模型已转换为FP8
    model._is_fp8 = True


def spmm_fp8(sparse_matrix, dense_matrix):
    """
    纯FP8格式的稀疏矩阵乘法：sparse_matrix @ dense_matrix
    
    手动实现，全程使用FP8进行计算（不转换到FP32）
    使用逐元素计算以保持FP8精度
    
    Args:
        sparse_matrix: 稀疏矩阵 (COO格式)
        dense_matrix: 稠密矩阵 (FP8格式)
    
    Returns:
        结果矩阵 (FP8格式)
    """
    # 检查dense_matrix是否是FP8格式
    is_fp8_input = dense_matrix.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    
    if not is_fp8_input:
        # 如果输入不是FP8，使用标准方法
        return torch.spmm(sparse_matrix, dense_matrix)
    
    # 手动实现纯FP8稀疏矩阵乘法
    if not sparse_matrix.is_sparse:
        sparse_matrix = sparse_matrix.to_sparse()
    
    # 获取稀疏矩阵的索引和值
    indices = sparse_matrix._indices()  # [2, nnz]
    values = sparse_matrix._values()     # [nnz]
    m, k = sparse_matrix.size()
    k2, n = dense_matrix.size()
    
    assert k == k2, f"矩阵维度不匹配: {k} != {k2}"
    
    # 将稀疏矩阵的值也转换为FP8（如果还不是）
    if values.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        values = convert_tensor_to_fp8(values, use_e4m3=True)
    
    # 初始化输出矩阵为FP8格式的零矩阵
    output = torch.zeros(m, n, dtype=torch.float8_e4m3fn, device=dense_matrix.device)
    
    # 手动计算稀疏矩阵乘法 - 纯FP8计算
    # result[i, j] = sum_k (sparse[i, k] * dense[k, j])
    row_indices = indices[0]  # 行索引
    col_indices = indices[1]  # 列索引
    
    # 使用scatter_add进行高效的稀疏乘法累加
    # 由于FP8不支持scatter_add，我们需要手动循环
    for idx in range(len(values)):
        i = row_indices[idx].item()
        k = col_indices[idx].item()
        
        # 纯FP8乘法：sparse_val (FP8) * dense_row (FP8)
        sparse_val = values[idx]
        dense_row = dense_matrix[k]
        
        # FP8乘法并累加到输出
        # 注意：这里的乘法和加法都在FP8下进行
        product = sparse_val * dense_row  # FP8 * FP8 -> FP8
        output[i] = output[i] + product    # FP8 + FP8 -> FP8
    
    return output


def mm_fp8(matrix1, matrix2):
    """
    纯FP8格式的稠密矩阵乘法：matrix1 @ matrix2
    
    手动实现，全程使用FP8进行计算（不转换到FP32）
    使用分块矩阵乘法以利用GPU并行性
    
    Args:
        matrix1: 第一个矩阵 (可能是FP8格式) [m, k]
        matrix2: 第二个矩阵 (可能是FP8格式) [k, n]
    
    Returns:
        结果矩阵 (FP8格式如果输入是FP8) [m, n]
    """
    # 检查输入是否是FP8格式
    is_fp8_1 = matrix1.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    is_fp8_2 = matrix2.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    
    if not (is_fp8_1 or is_fp8_2):
        # 都不是FP8，直接使用标准PyTorch矩阵乘法
        return torch.mm(matrix1, matrix2)
    
    # 确保两个矩阵都是FP8格式
    if not is_fp8_1:
        matrix1 = convert_tensor_to_fp8(matrix1, use_e4m3=True)
    if not is_fp8_2:
        matrix2 = convert_tensor_to_fp8(matrix2, use_e4m3=True)
    
    # 纯FP8矩阵乘法实现
    # C[i,j] = sum_k(A[i,k] * B[k,j])
    m, k = matrix1.size()
    k2, n = matrix2.size()
    
    assert k == k2, f"矩阵维度不匹配: {k} != {k2}"
    
    # 使用PyTorch的张量运算实现矩阵乘法（在FP8下）
    # 方法：利用broadcasting和求和
    # C = sum_k(A[:, k:k+1] * B[k:k+1, :])
    
    # 初始化输出矩阵为FP8零矩阵
    output = torch.zeros(m, n, dtype=torch.float8_e4m3fn, device=matrix1.device)
    
    # 逐列累加以减少内存使用
    # 这里我们利用PyTorch的广播机制在FP8下进行计算
    for k_idx in range(k):
        # A的第k列 [m, 1]
        col_a = matrix1[:, k_idx:k_idx+1]
        # B的第k行 [1, n]
        row_b = matrix2[k_idx:k_idx+1, :]
        
        # 外积：[m, 1] * [1, n] -> [m, n] (在FP8下计算)
        product = col_a * row_b
        
        # 累加到输出 (在FP8下累加)
        output = output + product
    
    return output


def delta_epsilon_activation(x, epsilon, c):
    """
    分段激活函数 δ_ε(x)
    
    δ_ε(x) = {
        0,                               if x ∈ [0, ε]
        θ * (1 + e^(-c)) / (1 + e^(-cθ)), if x ∈ (ε, 1]
    }
    
    其中 θ = (x - ε) / (1 - ε)
    
    Args:
        x: 输入张量
        epsilon: 阈值参数
        c: sigmoid参数
    
    Returns:
        激活后的张量
    """
    # 创建mask：x <= epsilon的位置为0
    mask = (x > epsilon).float()
    
    # 计算θ = (x - ε) / (1 - ε)
    theta = (x - epsilon) / (1 - epsilon + 1e-8)
    theta = torch.clamp(theta, 0, 1)  # 确保θ在[0,1]范围内
    
    # 计算激活值：θ * (1 + e^(-c)) / (1 + e^(-cθ))
    # 使用tensor的device和dtype确保在正确的设备上
    numerator = 1 + torch.exp(torch.tensor(-c, device=x.device, dtype=x.dtype))
    denominator = 1 + torch.exp(-c * theta)
    activation = theta * numerator / denominator
    
    # 应用mask
    output = mask * activation
    
    return output


def three_segment_activation(x, x1, x2, c):
    """
    三段激活函数 δ_ε(x)
    
    δ_ε(x) = {
        0,                                if x ∈ [0, x1]
        θ * (1 + e^(-c)) / (1 + e^(-cθ)), if x ∈ (x1, x2]
        1,                                if x ∈ (x2, 1]
    }
    
    其中 θ = (x - x1) / (x2 - x1)
    
    Args:
        x: 输入张量
        x1: 第一个阈值参数
        x2: 第二个阈值参数
        c: sigmoid参数
    
    Returns:
        激活后的张量
    """
    # 创建三个区间的mask
    mask1 = (x <= x1).float()  # [0, x1]
    mask2 = ((x > x1) & (x <= x2)).float()  # (x1, x2]
    mask3 = (x > x2).float()  # (x2, 1]
    
    # 第一段: 输出0
    segment1 = torch.zeros_like(x)
    
    # 第二段: θ * (1 + e^(-c)) / (1 + e^(-cθ))
    theta = (x - x1) / (x2 - x1 + 1e-8)
    theta = torch.clamp(theta, 0, 1)  # 确保θ在[0,1]范围内
    numerator = 1 + torch.exp(torch.tensor(-c, device=x.device))
    denominator = 1 + torch.exp(-c * theta)
    segment2 = theta * numerator / denominator
    
    # 第三段: 输出1
    segment3 = torch.ones_like(x)
    
    # 组合三个段落
    output = mask1 * segment1 + mask2 * segment2 + mask3 * segment3
    
    return output


def piecewise_linear_activation(x, y1=0.05, epsilon=0.1):
    """
    分段线性激活函数

    三个段落:
    - [0, epsilon]: 从 (0,0) 到 (epsilon, y1) 的直线
    - [epsilon, 1-epsilon]: 连接两个拐点的直线，经过 (0.5, 0.5)
    - [1-epsilon, 1]: 从 (1-epsilon, y2) 到 (1, 1) 的直线

    约束: y1 + y2 = 1 以确保中间段经过 (0.5, 0.5)

    Args:
        x: 输入张量
        y1: 第一个拐点 (epsilon, y1) 的y坐标
        epsilon: 段落划分参数

    Returns:
        激活后的张量
    """
    # 强制约束: y2 = 1 - y1
    y2 = 1 - y1

    # 段落1: [0, epsilon], y = (y1/epsilon) * x
    mask1 = (x <= epsilon).float()
    segment1 = (y1 / epsilon) * x if epsilon > 0 else torch.zeros_like(x)

    # 段落2: [epsilon, 1-epsilon], y = y1 + slope2 * (x - epsilon)
    # slope2 = (y2 - y1) / (1 - 2*epsilon)
    mask2 = ((x > epsilon) & (x <= 1-epsilon)).float()
    slope2 = (y2 - y1) / (1 - 2*epsilon) if (1 - 2*epsilon) > 0 else 0
    segment2 = y1 + slope2 * (x - epsilon)
    
    # 段落3: [1-epsilon, 1], y = y2 + slope3 * (x - (1-epsilon))
    # slope3 = (1 - y2) / epsilon
    mask3 = (x > 1-epsilon).float()
    slope3 = (1 - y2) / epsilon if epsilon > 0 else 0
    segment3 = y2 + slope3 * (x - (1 - epsilon))
    
    # 组合三个段落
    output = mask1 * segment1 + mask2 * segment2 + mask3 * segment3
    
    return output


class GraphConvolution(nn.Module):
    """标准图卷积层"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        # 检查是否使用FP8推理
        is_fp8 = self.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        
        if is_fp8:
            # FP8推理：使用自定义的FP8矩阵乘法
            support = mm_fp8(input, self.weight)
            output = spmm_fp8(adj, support)
        else:
            # 标准推理
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ColumnStochasticGraphConvolution(nn.Module):
    """
    列随机/双随机图卷积层 - 权重矩阵每列和为1（列随机）或行列和都为1（双随机）
    
    初始化方法：
    1. 在[0,1]之间均匀采样 w_ij ~ Uniform(0, 1)
    2. 阈值过滤：w_ij = max(w_ij, threshold)
    3. 最大值限制：w_ij = min(w_ij, max_weight)
    4. 列归一化（每列除以该列的和）或双随机归一化
    """
    def __init__(self, in_features, out_features, bias=True, w_threshold=0.0, 
                 doubly_stochastic=False, stochastic_iterations=20):
        super(ColumnStochasticGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_threshold = w_threshold  # 权重阈值参数
        self.doubly_stochastic = doubly_stochastic  # 是否使用双随机矩阵
        self.stochastic_iterations = stochastic_iterations  # 双随机迭代次数
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        初始化为列随机或双随机矩阵
        使用均匀分布初始化，然后进行相应的归一化，避免exp导致的数值不稳定
        """
        # Step 1: 在[0,1]之间采样均匀分布
        self.weight.data.uniform_(0, 1)
        
        # Step 2: 根据配置选择列随机或双随机归一化
        if self.doubly_stochastic:
            project_doubly_stochastic(self.weight, threshold=self.w_threshold, 
                                     max_iterations=self.stochastic_iterations)
        else:
            project_column_stochastic(self.weight, threshold=self.w_threshold)
        
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, input, adj):
        # 检查是否使用FP8推理
        is_fp8 = self.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        
        if is_fp8:
            # FP8推理：使用自定义的FP8矩阵乘法
            support = mm_fp8(input, self.weight)
            output = spmm_fp8(adj, support)
        else:
            # 标准推理
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


def project_column_stochastic(weight, max_weight=0.05, threshold=0.0):
    """
    将权重矩阵投影到列随机约束（每列和为1，所有元素非负）
    使用阈值过滤 + 列归一化
    
    Args:
        weight: 权重矩阵
        max_weight: 单个权重的最大值，防止权重坍缩。默认0.05（约3倍均匀分布值1/64）
        threshold: 阈值参数，小于此值的权重将被置为0。默认0.0（相当于ReLU）
    
    使用in-place操作避免创建新tensor
    """
    with torch.no_grad():
        # Step 1: 使用阈值过滤（小于阈值的权重置为0）
        weight.clamp_(min=threshold)
        # 将小于阈值的值精确置为0
        weight[weight < threshold] = 0
        
        # Step 2: 限制单个权重的最大值，防止权重坍缩
        weight.clamp_(max=max_weight)
        
        # Step 3: 列归一化（每列除以列和，保证每列和为1）
        col_sums = weight.sum(dim=0, keepdim=True)  # [1, out_features]
        col_sums = torch.clamp(col_sums, min=1e-8)  # 避免除以0
        weight.div_(col_sums)


def project_doubly_stochastic(weight, max_weight=0.05, threshold=0.0, max_iterations=20):
    """
    将权重矩阵投影到双随机约束（每行和为1，每列和为1，所有元素非负）
    使用阈值过滤 + 交替行列归一化（Sinkhorn-Knopp算法的简化版本）
    
    Args:
        weight: 权重矩阵（必须是方阵）
        max_weight: 单个权重的最大值，防止权重坍缩。默认0.05
        threshold: 阈值参数，小于此值的权重将被置为0。默认0.0（相当于ReLU）
        max_iterations: 最大迭代次数，默认20
    
    使用in-place操作避免创建新tensor
    """
    with torch.no_grad():
        # Step 1: 使用阈值过滤（小于阈值的权重置为0）
        weight.clamp_(min=threshold)
        weight[weight < threshold] = 0
        
        # Step 2: 限制单个权重的最大值，防止权重坍缩
        weight.clamp_(max=max_weight)
        
        # Step 3: 交替进行行归一化和列归一化，迭代收敛到双随机矩阵
        for _ in range(max_iterations):
            # 列归一化
            col_sums = weight.sum(dim=0, keepdim=True)  # [1, out_features]
            col_sums = torch.clamp(col_sums, min=1e-8)
            weight.div_(col_sums)
            
            # 行归一化
            row_sums = weight.sum(dim=1, keepdim=True)  # [in_features, 1]
            row_sums = torch.clamp(row_sums, min=1e-8)
            weight.div_(row_sums)


def apply_identity_mixing(weight, eta):
    """
    对权重矩阵应用单位矩阵混合：W_l = eta * I + (1-eta) * W_l
    
    Args:
        weight: 权重矩阵 [d, d]（必须是方阵）
        eta: 单位矩阵混合系数，范围[0, 1]
    
    该操作保持列随机特性：
    - 如果 W_l 的每列和为1，则混合后仍为1
    - 证明：sum(eta*I[:, j] + (1-eta)*W[:, j]) = eta*1 + (1-eta)*1 = 1
    
    使用in-place操作避免创建新tensor
    """
    if eta <= 0 or eta >= 1:
        return  # eta=0时不需要混合，eta=1时变成纯单位矩阵（不合理）
    
    with torch.no_grad():
        # 创建单位矩阵
        d = weight.size(0)
        identity = torch.eye(d, device=weight.device, dtype=weight.dtype)
        
        # W_l = eta * I + (1-eta) * W_l
        weight.mul_(1 - eta)  # W_l = (1-eta) * W_l
        weight.add_(identity, alpha=eta)  # W_l = W_l + eta * I


class EnhancedGCNII(nn.Module):
    """
    Enhanced GCNII (Version 4.0)
    
    与标准GCNII的唯一区别：层更新公式多了线性传播项
    
    标准GCNII:
    H^{(l+1)} = σ(((1-α)*Â*H^{(l)} + α*H^{(0)})*((1-β_l)*I + β_l*W_{(l)}))
    
    Enhanced GCNII:
    H^{(l+1)} = Â*H^{(l)}*W'_{(l)} + σ(((1-α)*Â*H^{(l)} + α*H^{(0)})*((1-β_l)*I + β_l*W_{(l)}))
    
    关键特性:
    1. 每层有两个独立权重矩阵: W' (线性传播，列随机) 和 W (GCNII部分，标准权重)
    2. W' 是列随机矩阵（每列元素和为1）
       - 初始化：Uniform(0,1) → 列归一化
    3. H0预处理：H_0 = (1-γ)*H_0 + γ*1*c^T，其中c是可学习参数
    4. 其余完全和标准GCNII一致
    """
    def __init__(self, nfeat, nhid, nclass, nlayers, gamma=0.1, dropout=0.6, alpha=0.1, lambda_=0.5):
        super(EnhancedGCNII, self).__init__()
        self.nlayers = nlayers
        self.dropout = dropout
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma
        
        # 第一层：从输入特征到隐藏层
        self.fc_in = nn.Linear(nfeat, nhid)
        
        # 可学习向量c ∈ R^d，用于H0预处理
        self.c = nn.Parameter(torch.randn(nhid))
        
        # GCNII部分的卷积层 (W) - 标准权重
        self.convs_gcnii = nn.ModuleList([
            GraphConvolution(nhid, nhid) for _ in range(nlayers)
        ])
        
        # 线性传播部分的卷积层 (W') - 列随机权重
        self.convs_linear = nn.ModuleList([
            ColumnStochasticGraphConvolution(nhid, nhid) for _ in range(nlayers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(nhid, nclass)
    
    def normalize_adj(self, adj):
        """对称归一化邻接矩阵: Â = D^{-1/2}*A*D^{-1/2}"""
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # Add self-loops: A = A + I
        adj_with_self_loops = adj_dense + torch.eye(adj_dense.size(0), device=adj.device)
        
        # Compute degree matrix D
        degrees = adj_with_self_loops.sum(1)
        
        # Compute D^{-1/2}
        d_inv_sqrt = torch.pow(degrees, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Â = D^{-1/2} * A * D^{-1/2}
        # 注意：归一化邻接矩阵总是在FP32进行，不使用FP8（因为是稀疏矩阵且只计算一次）
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self_loops), d_mat_inv_sqrt)
        
        return adj_normalized.to_sparse()
    
    def forward(self, x, adj):
        """
        前向传播
        
        1. 初始变换: H^{(0)} = ReLU(X * W_in)
        2. H0预处理: H_0 = (1-γ)*H_0 + γ*1*c^T
        3. 逐层更新: H^{(l+1)} = Â*H^{(l)}*W'_{(l)} + σ(((1-α)*Â*H^{(l)} + α*H^{(0)})*((1-β_l)*I + β_l*W_{(l)}))
        4. 输出: H^{(L)} * W_out
        """
        # 检查是否使用FP8推理
        is_fp8 = hasattr(self, '_is_fp8') and self._is_fp8
        
        # 归一化邻接矩阵
        adj_normalized = self.normalize_adj(adj)
        
        # Step 1: 初始变换 X -> H^{(0)}
        x = self.fc_in(x)  # [N, nfeat] -> [N, nhid]
        x = F.relu(x)
        
        # Step 2: H0预处理 H_0 = (1-γ)*H_0 + γ*1*c^T
        # 其中 1*c^T 将可学习向量c广播到所有节点
        # c: [nhid], 广播为 [N, nhid]
        c_broadcast = self.c.unsqueeze(0).expand(x.size(0), -1)  # [1, nhid] -> [N, nhid]
        x = (1 - self.gamma) * x + self.gamma * c_broadcast
        
        # 存储预处理后的H^{(0)}用于GCNII的初始残差连接
        h0 = x
        h = x
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 逐层传播
        for i in range(self.nlayers):
            # 计算β_l (层依赖参数)
            beta = math.log(self.lambda_ / (i + 1) + 1)
            
            # 线性传播项: Â*H^{(l)}*W'_{(l)}
            linear_part = self.convs_linear[i](h, adj_normalized)
            
            # GCNII部分:
            # Step 1: 计算 (1-α)*Â*H^{(l)} + α*H̃^{(0)}
            if is_fp8:
                h_propagated = spmm_fp8(adj_normalized, h)
            else:
                h_propagated = torch.spmm(adj_normalized, h)
            h_mixed = (1 - self.alpha) * h_propagated + self.alpha * h0
            
            # Step 2: 计算 W_{(l)} 部分 (使用 GraphConvolution)
            # GraphConvolution 实现的是 Â*(input*W)
            # 我们需要的是 (input)*((1-β_l)*I + β_l*W)
            # 因此需要手动实现
            w = self.convs_gcnii[i].weight
            identity = torch.eye(w.size(0), w.size(1), device=w.device, dtype=w.dtype)
            w_mixed = (1 - beta) * identity + beta * w
            if is_fp8:
                h_gcnii = mm_fp8(h_mixed, w_mixed)
            else:
                h_gcnii = torch.mm(h_mixed, w_mixed)
            
            # 添加bias (如果有)
            if self.convs_gcnii[i].bias is not None:
                h_gcnii = h_gcnii + self.convs_gcnii[i].bias
            
            # Step 3: 激活函数
            h_gcnii = F.relu(h_gcnii)
            
            # 组合: H^{(l+1)} = 线性项 + GCNII项
            h = linear_part + h_gcnii
            h = F.dropout(h, self.dropout, training=self.training)
        
        # 输出层
        h = self.fc_out(h)
        
        return F.log_softmax(h, dim=1)


class StandardGCNII(nn.Module):
    """
    标准GCNII (基线模型)
    
    更新公式:
    H^{(l+1)} = σ(((1-α)*Â*H^{(l)} + α*H^{(0)})*((1-β_l)*I + β_l*W_{(l)}))
    """
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout=0.6, alpha=0.1, lambda_=0.5, 
                 weight_sharing=False, sharing_interval=2):
        super(StandardGCNII, self).__init__()
        self.nlayers = nlayers
        self.dropout = dropout
        self.alpha = alpha
        self.lambda_ = lambda_
        self.weight_sharing = weight_sharing
        self.sharing_interval = sharing_interval
        
        # 初始变换层
        self.fc_in = nn.Linear(nfeat, nhid)
        
        # GCNII层
        if weight_sharing and nlayers > 0:
            # 权重共享模式：每 sharing_interval 层共享权重
            num_groups = (nlayers + sharing_interval - 1) // sharing_interval
            # 只创建基础层，不创建重复引用
            self.convs = nn.ModuleList([GraphConvolution(nhid, nhid) for _ in range(num_groups)])
            # 存储层到组的映射
            self.layer_to_group = [i // sharing_interval for i in range(nlayers)]
        else:
            # 标准模式
            self.convs = nn.ModuleList([
                GraphConvolution(nhid, nhid) for _ in range(nlayers)
            ])
            self.layer_to_group = None
        
        # 输出层
        self.fc_out = nn.Linear(nhid, nclass)
    
    def normalize_adj(self, adj):
        """对称归一化邻接矩阵"""
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        adj_with_self_loops = adj_dense + torch.eye(adj_dense.size(0), device=adj.device)
        degrees = adj_with_self_loops.sum(1)
        d_inv_sqrt = torch.pow(degrees, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self_loops), d_mat_inv_sqrt)
        
        return adj_normalized.to_sparse()
    
    def forward(self, x, adj):
        """标准GCNII前向传播"""
        # 检查是否使用FP8推理
        is_fp8 = hasattr(self, '_is_fp8') and self._is_fp8
        
        adj_normalized = self.normalize_adj(adj)
        
        # 初始变换
        x = F.relu(self.fc_in(x))
        x = F.dropout(x, self.dropout, training=self.training)
        
        h0 = x
        h = x
        
        # GCNII层
        for i in range(self.nlayers):
            beta = math.log(self.lambda_ / (i + 1) + 1)
            
            # (1-α)*Â*H^{(l)} + α*H^{(0)}
            if is_fp8:
                h_propagated = spmm_fp8(adj_normalized, h)
            else:
                h_propagated = torch.spmm(adj_normalized, h)
            h_mixed = (1 - self.alpha) * h_propagated + self.alpha * h0
            
            # ((1-β_l)*I + β_l*W_{(l)})
            # 在权重共享模式下，使用layer_to_group映射来选择正确的卷积层
            conv_idx = self.layer_to_group[i] if self.layer_to_group is not None else i
            w = self.convs[conv_idx].weight
            # 使用类型和设备匹配的identity，确保梯度能正确传播
            identity = torch.eye(w.size(0), w.size(1), dtype=w.dtype, device=w.device)
            w_mixed = (1 - beta) * identity + beta * w
            if is_fp8:
                h = mm_fp8(h_mixed, w_mixed)
            else:
                h = torch.mm(h_mixed, w_mixed)
            
            if self.convs[conv_idx].bias is not None:
                h = h + self.convs[conv_idx].bias
            
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        
        # 输出
        h = self.fc_out(h)
        return F.log_softmax(h, dim=1)


class NewActivationGNN(nn.Module):
    """
    新激活函数模型 Version 6.1（使用三段激活函数）
    
    更新公式:
    X_0 = normalize((1-γ)*H^{(0)} + γ*1*ReLU(c)^T)  # 仅在预处理后归一化一次到[0,1]
    X_{l+1} = δ_ε(D^{-1}AX_lW_l)  # 三段激活函数，直接作用于图卷积结果
    β_l = min(0.5, (l+1)/L * 0.5)  # 残差权重随层数线性增长
    X_{l+1} = (1-β_l)*X_{l+1} + β_l*X_0  # 每层都有初始残差连接
    
    其中:
    - W_l是列随机矩阵（每列和为1），初始化为[0,1]均匀采样后归一化
    - 单个权重≤0.05（防止权重坍缩）
    - 使用D^{-1}A进行行归一化（加自环）
    - δ_ε是三段激活函数（three_segment activation）
    - H0预处理: H_0 = (1-γ)*H^{(0)} + γ*1*ReLU(c)^T，然后归一化到[0,1]
    
    关键改进（解决深层信号衰减）:
    1. **X只在预处理后归一化一次**：确保初始输入在[0,1]范围内
    2. **三段激活函数**：δ_ε(x)在[0,x1]输出0，在(x1,x2]平滑过渡，在(x2,1]输出1
    3. **自适应残差连接**：每层都有，权重从0线性增长到0.5
       - 浅层(β≈0)：主要学习新特征
       - 深层(β≈0.5)：50%保持初始特征，防止过度平滑
    4. **列随机权重约束**：训练后投影，确保每列和为1，增强稳定性
    5. 权重上界max_weight=0.05防止权重分布坍缩
    6. 梯度裁剪(max_norm=1.0)防止权重更新过大
    """
    def __init__(self, nfeat, nhid, nclass, nlayers, gamma=0.3, alpha=0.5, epsilon=0.1, 
                 c_activation=-1.0, dropout=0.1, verbose=False, w_threshold=0.0, 
                 activation_type='delta_epsilon', y1=0.05, progressive_alpha=False,
                 x1=0.0, x2=1.0, weight_sharing=False, sharing_interval=2,
                 doubly_stochastic=False, stochastic_iterations=20):
        super(NewActivationGNN, self).__init__()
        self.nlayers = nlayers
        self.dropout = dropout
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.c_activation = c_activation
        self.verbose = verbose  # 是否输出详细统计信息
        self._should_print = False  # 是否应该打印统计信息（由外部控制）
        self.w_threshold = w_threshold  # 权重阈值参数
        self.activation_type = activation_type  # 激活函数类型: 'delta_epsilon', 'piecewise_linear', or 'three_segment'
        self.y1 = y1  # 分段线性激活函数的y1参数
        self.progressive_alpha = progressive_alpha  # 是否使用渐进alpha
        self.x1 = x1  # 三段激活函数的第一个阈值
        self.x2 = x2  # 三段激活函数的第二个阈值
        self.weight_sharing = weight_sharing  # 是否启用权重共享
        self.sharing_interval = sharing_interval  # 权重共享间隔
        self.doubly_stochastic = doubly_stochastic  # 是否使用双随机矩阵
        self.stochastic_iterations = stochastic_iterations  # 双随机迭代次数
        
        # 第一层：从输入特征到隐藏层
        self.fc_in = nn.Linear(nfeat, nhid)
        
        # 可学习向量c ∈ R^d，用于H0预处理（初始化为较大正值，抬高初始特征）
        # 使用uniform(0.8, 1.2)初始化，确保ReLU后仍有较大值
        self.c = nn.Parameter(torch.empty(nhid).uniform_(0.8, 1.2))
        
        # 每层的列随机/双随机权重矩阵
        if weight_sharing and nlayers > 0:
            # 权重共享模式：每 sharing_interval 层共享权重
            num_groups = (nlayers + sharing_interval - 1) // sharing_interval
            # 只创建基础层，不创建重复引用
            self.convs = nn.ModuleList([ColumnStochasticGraphConvolution(
                nhid, nhid, w_threshold=w_threshold,
                doubly_stochastic=doubly_stochastic, 
                stochastic_iterations=stochastic_iterations) 
                for _ in range(num_groups)])
            # 存储层到组的映射
            self.layer_to_group = [i // sharing_interval for i in range(nlayers)]
        else:
            # 标准模式
            self.convs = nn.ModuleList([
                ColumnStochasticGraphConvolution(
                    nhid, nhid, w_threshold=w_threshold,
                    doubly_stochastic=doubly_stochastic,
                    stochastic_iterations=stochastic_iterations) 
                for _ in range(nlayers)
            ])
            self.layer_to_group = None
        
        # 输出层
        self.fc_out = nn.Linear(nhid, nclass)
    
    def enable_print_stats(self):
        """启用统计信息打印"""
        self._should_print = True
    
    def disable_print_stats(self):
        """禁用统计信息打印"""
        self._should_print = False
    
    def print_init_stats(self):
        """打印初始化统计信息"""
        if self.verbose:
            print("\n" + "="*80)
            print("NewActivationGNN 初始化统计")
            print("="*80)
            c_positive = F.relu(self.c)
            print(f"可学习向量c (原始): 形状={self.c.shape}")
            print(f"  范围: [{self.c.min().item():.6f}, {self.c.max().item():.6f}]")
            print(f"  均值: {self.c.mean().item():.6f}")
            print(f"可学习向量c (ReLU后，实际使用): 形状={c_positive.shape}")
            print(f"  范围: [{c_positive.min().item():.6f}, {c_positive.max().item():.6f}]")
            print(f"  均值: {c_positive.mean().item():.6f}")
            print(f"\n权重矩阵W初始化 (列随机):")
            for i, conv in enumerate(self.convs):
                w = conv.weight
                print(f"Layer {i}: 形状={w.shape}")
                print(f"  范围: [{w.min().item():.6f}, {w.max().item():.6f}]")
                print(f"  均值: {w.mean().item():.6f}")
                col_sums = w.sum(dim=0)
                print(f"  列和范围: [{col_sums.min().item():.6f}, {col_sums.max().item():.6f}]")
            print("="*80 + "\n")
    
    def normalize_adj_row(self, adj):
        """行归一化邻接矩阵: D^{-1}A"""
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # Add self-loops: A = A + I
        adj_with_self_loops = adj_dense + torch.eye(adj_dense.size(0), device=adj.device)
        
        # Compute degree matrix D
        degrees = adj_with_self_loops.sum(1)
        
        # Compute D^{-1}
        d_inv = torch.pow(degrees, -1.0)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        
        # D^{-1}A
        adj_normalized = torch.mm(d_mat_inv, adj_with_self_loops)
        
        return adj_normalized.to_sparse()
    
    def forward(self, x, adj):
        """
        前向传播 Version 6.1（使用三段激活函数）
        
        1. 初始变换: H^{(0)} = ReLU(X * W_in)
        2. H0预处理: H_0 = (1-γ)*H^{(0)} + γ*1*ReLU(c)^T
        3. **归一化到[0,1]**: H_0 = (H_0 - min(H_0)) / (max(H_0) - min(H_0))  # 只归一化一次
        4. 逐层更新:
           a. z = D^{-1}AX_lW_l  # 图卷积传播
           b. X_{l+1} = δ_ε(z)  # 三段激活函数
           c. β_l = min(0.5, (l+1)/L * 0.5)  # 残差权重
           d. X_{l+1} = (1-β_l)*X_{l+1} + β_l*H_0  # 自适应残差连接
           e. X_{l+1} = dropout(X_{l+1})  # 仅训练时
        5. 输出: log_softmax(X_L * W_out)
        
        关键特点：
        - X只在预处理后归一化一次到[0,1]范围
        - 三段激活函数直接作用于图卷积结果，无激活函数混合
        - 自适应残差连接防止深层过平滑
        - 列随机权重约束增强训练稳定性
        """
        # 行归一化邻接矩阵 D^{-1}A
        adj_normalized = self.normalize_adj_row(adj)
        
        # Step 1: 初始变换 X -> H^{(0)}
        x = self.fc_in(x)
        x = F.relu(x)
        
        # Step 2: H0预处理 H_0 = (1-γ)*H_0 + γ*1*c^T
        # 对c使用ReLU确保非负，避免引入负值
        c_positive = F.relu(self.c)
        c_broadcast = c_positive.unsqueeze(0).expand(x.size(0), -1)
        x = (1 - self.gamma) * x + self.gamma * c_broadcast
        
        # 打印H0预处理后的统计（只打印一次，因为c之后不再使用）
        if self.verbose and self.training and self._should_print:
            print(f"\n{'='*80}")
            print(f"H0预处理统计（训练中的c值）")
            print(f"{'='*80}")
            print(f"可学习向量c (ReLU后): 形状={c_positive.shape}")
            print(f"  范围: [{c_positive.min().item():.6f}, {c_positive.max().item():.6f}]")
            print(f"  均值: {c_positive.mean().item():.6f}")
            print(f"gamma权重: {self.gamma}")
            print(f"H0 (预处理前): 均值={((x - self.gamma * c_broadcast) / (1 - self.gamma)).mean().item():.6f}")
            print(f"H0 (预处理后，归一化前): 均值={x.mean().item():.6f}")
            print(f"  范围: [{x.min().item():.6f}, {x.max().item():.6f}]")
            print(f"c的贡献: {(self.gamma * c_broadcast).mean().item():.6f}")
        
        # Step 3: 归一化X到[0,1]范围（只在预处理后归一化一次）
        x_min = x.min()
        x_max = x.max()
        x_range = x_max - x_min
        if x_range > 1e-8:
            x = (x - x_min) / x_range
        else:
            x = torch.clamp(x, 0, 1)  # 退化情况
        
        if self.verbose and self.training and self._should_print:
            print(f"H0 (归一化后): 均值={x.mean().item():.6f}")
            print(f"  范围: [{x.min().item():.6f}, {x.max().item():.6f}]")
            print(f"{'='*80}")
        
        # 保存初始H0
        h = x
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 保存初始H0用于残差连接
        h0_residual = h
        
        # 逐层传播
        for i in range(self.nlayers):
            # 根据外部设置决定是否输出统计信息
            show_stats = self.verbose and self.training and self._should_print
            
            if show_stats:
                print(f"\n{'='*80}")
                print(f"Layer {i} - 前向传播统计")
                print(f"{'='*80}")
                print(f"输入X_l: 形状={h.shape}")
                print(f"  范围: [{h.min().item():.6f}, {h.max().item():.6f}]")
                print(f"  均值: {h.mean().item():.6f}")
                
                # 获取当前层的卷积层（考虑权重共享）
                conv_idx = self.layer_to_group[i] if self.layer_to_group is not None else i
                w = self.convs[conv_idx].weight
                print(f"权重W_l: 形状={w.shape}")
                print(f"  范围: [{w.min().item():.6f}, {w.max().item():.6f}]")
                print(f"  均值: {w.mean().item():.6f}")
                col_sums = w.sum(dim=0)
                print(f"  列和范围: [{col_sums.min().item():.6f}, {col_sums.max().item():.6f}]")
            
            # Step 1: 计算 D^{-1}AX_lW_l（不做任何处理）
            # 在权重共享模式下，使用layer_to_group映射来选择正确的卷积层
            conv_idx = self.layer_to_group[i] if self.layer_to_group is not None else i
            z = self.convs[conv_idx](h, adj_normalized)
            
            if show_stats:
                print(f"\nD^{{-1}}AX_lW_l:")
                print(f"  范围: [{z.min().item():.6f}, {z.max().item():.6f}]")
                print(f"  均值: {z.mean().item():.6f}")
            
            # Step 2: 应用激活函数
            if self.activation_type == 'piecewise_linear':
                z_activated = piecewise_linear_activation(z, self.y1, self.epsilon)
                activation_name = f"piecewise_linear(z, y1={self.y1})"
            elif self.activation_type == 'three_segment':
                z_activated = three_segment_activation(z, self.x1, self.x2, self.c_activation)
                activation_name = f"three_segment(z, x1={self.x1}, x2={self.x2}, c={self.c_activation})"
            else:  # delta_epsilon
                z_activated = delta_epsilon_activation(z, self.epsilon, self.c_activation)
                activation_name = "δ_ε(z)"
            
            if show_stats:
                print(f"{activation_name} 范围: [{z_activated.min().item():.6f}, {z_activated.max().item():.6f}]")
            
            # Step 3: 根据激活函数类型决定更新方式
            if self.activation_type == 'three_segment':
                # 三段激活函数: X_{l+1} = δ_ε(D^{-1}AX_lW_l)
                h_new = z_activated
            else:
                # 其他激活函数: X_{l+1} = (1-α)*z + α*δ_ε(z)
                # 如果使用渐进alpha，则alpha从初始值逐层递增到1
                if self.progressive_alpha:
                    # alpha从self.alpha线性增长到1.0
                    # 第0层: alpha = self.alpha
                    # 最后一层(nlayers-1): alpha = 1.0
                    if self.nlayers > 1:
                        layer_alpha = self.alpha + (1.0 - self.alpha) * i / (self.nlayers - 1)
                    else:
                        layer_alpha = self.alpha
                else:
                    layer_alpha = self.alpha
                
                h_new = (1 - layer_alpha) * z + layer_alpha * z_activated
            
            # 每层都添加初始残差连接，权重随层数增加（类似GCNII的initial residual）
            # 深层网络需要更强的残差连接来防止信号衰减
            beta_residual = min(0.5, (i + 1) / self.nlayers * 0.5)  # 0到0.5线性增长
            h = (1 - beta_residual) * h_new + beta_residual * h0_residual
            
            if show_stats:
                if self.progressive_alpha:
                    print(f"层{i}的alpha={layer_alpha:.4f} (渐进模式)")
                else:
                    print(f"alpha={layer_alpha:.4f} (固定模式)")
                print(f"残差权重β={beta_residual:.4f}")
                print(f"输出X_{{l+1}} (dropout前): 范围=[{h.min().item():.6f}, {h.max().item():.6f}], 均值={h.mean().item():.6f}")
            
            # Dropout
            h = F.dropout(h, self.dropout, training=self.training)
            
            if show_stats:
                print(f"输出X_{{l+1}} (dropout后): 范围=[{h.min().item():.6f}, {h.max().item():.6f}], 均值={h.mean().item():.6f}")
                print(f"{'='*80}")
        
        # 输出层
        h = self.fc_out(h)
        
        return F.log_softmax(h, dim=1)
