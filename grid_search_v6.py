"""
Version 4.0 Grid Search - 训练时归一化，推理时不归一化
基于Version 3.0，模型使用models_v5（简化归一化策略：X用clamp，W用ReLU+列归一化）
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models_v6 import EnhancedGCNII, StandardGCNII, NewActivationGNN
from datasets_v5 import load_data
import os
import sys
import random
import gc
import time
import shutil
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from torch_geometric.nn import GCNConv, GATConv, APPNP as PyG_APPNP, SGConv

# Intel Neural Compressor相关导入（用于INT8量化推理）
try:
    from inc_quantization import quantize_model_to_int8, evaluate_int8_model, INC_AVAILABLE
    if not INC_AVAILABLE:
        print("警告: Intel Neural Compressor未安装，INT8推理功能不可用。")
        print("请安装: pip install neural-compressor")
except ImportError:
    INC_AVAILABLE = False
    print("警告: 量化模块未找到，INT8推理功能不可用。")
    print("请确保inc_quantization.py文件存在")

# ONNX Runtime相关导入（已弃用，保留以防需要）
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['mathtext.fontset'] = 'dejavusans'
rcParams['axes.linewidth'] = 1.2
rcParams['axes.labelsize'] = 13
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = '#f9f9f9'

# 定义色系 - 高级渐进色
_COLOR_PALETTE = {
    'lyapunov': '#1e5670',      # 深青蓝色
    'gcn': '#6bb3c0',           # 中等青绿色
    'gat': '#c4e5ef'            # 浅青蓝色
}

_LABEL_CONFIG = {
    'lyapunov': 'LyapunovGCN',
    'gcn': 'GCN',
    'gat': 'GAT'
}


class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()




def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def compute_metrics(output, labels, return_dict=False):
    """计算多个评价指标 - GNN节点分类标准指标"""
    preds = output.max(1)[1].cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    acc = accuracy(output, labels).item()
    macro_f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
    macro_precision = precision_score(labels_np, preds, average='macro', zero_division=0)
    macro_recall = recall_score(labels_np, preds, average='macro', zero_division=0)
    
    if return_dict:
        return {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall
        }
    return acc, macro_f1, macro_precision, macro_recall


# ======================== Backbone Models ========================

class GCN(torch.nn.Module):
    """GCN (Kipf & Welling, ICLR 2017)"""
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout=0.5, weight_sharing=False, sharing_interval=2):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.weight_sharing = weight_sharing
        self.sharing_interval = sharing_interval
        
        if weight_sharing and nlayers > 1:
            # 权重共享模式：每 sharing_interval 层共享权重
            # 只创建必要的层数
            self.convs.append(GCNConv(nfeat, nhid))  # 第一层
            
            # 计算中间层需要多少组共享权重
            num_groups = (nlayers - 2 + sharing_interval - 1) // sharing_interval
            for _ in range(num_groups):
                self.convs.append(GCNConv(nhid, nhid))
            
            self.convs.append(GCNConv(nhid, nclass))  # 最后一层
            
            # 创建层到组的映射：第一层和最后一层独立，中间层共享
            self.layer_to_group = [0]  # 第一层
            for i in range(nlayers - 2):
                group_idx = i // sharing_interval
                self.layer_to_group.append(1 + group_idx)
            self.layer_to_group.append(len(self.convs) - 1)  # 最后一层
        else:
            # 标准模式：每层独立权重
            self.convs.append(GCNConv(nfeat, nhid))
            for _ in range(nlayers - 2):
                self.convs.append(GCNConv(nhid, nhid))
            self.convs.append(GCNConv(nhid, nclass))
            self.layer_to_group = None
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        num_layers = len(self.layer_to_group) if self.layer_to_group is not None else len(self.convs)
        for i in range(num_layers - 1):
            conv_idx = self.layer_to_group[i] if self.layer_to_group is not None else i
            x = self.convs[conv_idx](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层
        last_idx = self.layer_to_group[-1] if self.layer_to_group is not None else len(self.convs) - 1
        x = self.convs[last_idx](x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """GAT (Veličković et al., ICLR 2018)"""
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout=0.5, heads=8, weight_sharing=False, sharing_interval=2):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.weight_sharing = weight_sharing
        self.sharing_interval = sharing_interval
        
        if weight_sharing and nlayers > 1:
            # 权重共享模式
            # 只创建必要的层数
            self.convs.append(GATConv(nfeat, nhid, heads=heads, dropout=dropout))  # 第一层
            
            num_groups = (nlayers - 2 + sharing_interval - 1) // sharing_interval
            for _ in range(num_groups):
                self.convs.append(GATConv(nhid * heads, nhid, heads=heads, dropout=dropout))
            
            self.convs.append(GATConv(nhid * heads, nclass, heads=1, concat=False, dropout=dropout))  # 最后一层
            
            # 创建层到组的映射：第一层和最后一层独立，中间层共享
            self.layer_to_group = [0]  # 第一层
            for i in range(nlayers - 2):
                group_idx = i // sharing_interval
                self.layer_to_group.append(1 + group_idx)
            self.layer_to_group.append(len(self.convs) - 1)  # 最后一层
        else:
            # 标准模式
            self.convs.append(GATConv(nfeat, nhid, heads=heads, dropout=dropout))
            for _ in range(nlayers - 2):
                self.convs.append(GATConv(nhid * heads, nhid, heads=heads, dropout=dropout))
            self.convs.append(GATConv(nhid * heads, nclass, heads=1, concat=False, dropout=dropout))
            self.layer_to_group = None
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        num_layers = len(self.layer_to_group) if self.layer_to_group is not None else len(self.convs)
        for i in range(num_layers - 1):
            conv_idx = self.layer_to_group[i] if self.layer_to_group is not None else i
            x = self.convs[conv_idx](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层
        last_idx = self.layer_to_group[-1] if self.layer_to_group is not None else len(self.convs) - 1
        x = self.convs[last_idx](x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP(torch.nn.Module):
    """APPNP (Klicpera et al., ICLR 2019)"""
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout=0.5, alpha=0.1, K=10):
        super(APPNP, self).__init__()
        self.lin1 = torch.nn.Linear(nfeat, nhid)
        self.lin2 = torch.nn.Linear(nhid, nclass)
        self.prop = PyG_APPNP(K=K, alpha=alpha, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    """SGC (Wu et al., ICML 2019)"""
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout=0.5, K=2):
        super(SGC, self).__init__()
        self.conv = SGConv(nfeat, nclass, K=K, cached=True)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_epoch(model, optimizer, features, adj, labels, idx_train, project_weights=False, clip_grad=True, eta=0.0, w_threshold=0.0, track_gradients=False, lr=0.01, epoch=0):
    """
    训练一个epoch，可选地追踪第一层权重梯度的统计信息
    
    Returns:
        如果track_gradients=False: (loss, acc)
        如果track_gradients=True: (loss, acc, gradient_stats)
            gradient_stats是一个字典，包含：
            - 'variance': 梯度方差
            - 'norm': 梯度L2范数
            - 'mean_abs': 梯度绝对值的均值
    """
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.nll_loss(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    
    # 梯度裁剪：防止梯度爆炸（必须在计算梯度方差之前，以反映实际用于更新的梯度）
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 计算梯度统计（在梯度裁剪之后，以反映实际用于权重更新的梯度稳定性）
    normalized_grad_variance = None
    first_layer_grad_var = None
    last_layer_grad_var = None
    
    if track_gradients:
        # 计算所有GCN层权重的梯度方差，同时记录第一层和最后一层
        grad_variances = []
        grad_details = []  # 用于调试
        first_layer_var = None
        last_layer_var = None
        
        for name, param in model.named_parameters():
            # 跳过没有梯度的参数
            if param.grad is None:
                continue
            # 只看convs层的weight参数（不是bias，也不是fc_in/fc_out）
            if 'weight' not in name or 'bias' in name:
                continue
            if 'fc_in' in name or 'fc_out' in name:
                continue
            # 只看convs层或convs_linear层
            if 'convs.' in name or 'convs_linear.' in name:
                grad = param.grad.data
                var = grad.var().item()
                grad_variances.append(var)
                
                # 记录第一层和最后一层
                if 'convs.0.' in name or 'convs_linear.0.' in name:
                    first_layer_var = var
                # 假设最后一层的索引最大
                layer_idx = int(name.split('.')[1])
                if last_layer_var is None or layer_idx > int(grad_details[-1]['name'].split('.')[1]) if grad_details else True:
                    last_layer_var = var
                
                # 记录每层的梯度信息（用于调试）
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                grad_mean = grad.mean().item()
                grad_details.append({
                    'name': name,
                    'var': var,
                    'range': (grad_min, grad_max),
                    'mean': grad_mean,
                    'layer_idx': layer_idx
                })

        if grad_variances:
            # 平均梯度方差
            raw_variance = np.mean(grad_variances)
            normalized_grad_variance = raw_variance / (lr ** 2) if lr > 0 else raw_variance
            
            # 第一层和最后一层的归一化梯度方差
            if first_layer_var is not None:
                first_layer_grad_var = first_layer_var / (lr ** 2) if lr > 0 else first_layer_var
            if last_layer_var is not None:
                last_layer_grad_var = last_layer_var / (lr ** 2) if lr > 0 else last_layer_var
            
            # 调试：打印详细的梯度信息（仅第一个epoch）
            debug_gradients = (epoch == 0)  # 只在第一个epoch打印
            if debug_gradients and len(grad_details) > 0:
                print(f"\n[Grad Debug] Found {len(grad_variances)} GCN layers:")
                # 打印第一层
                first_detail = grad_details[0]
                print(f"  FIRST: {first_detail['name']}: var={first_detail['var']:.6e}, "
                      f"range=[{first_detail['range'][0]:.6e}, {first_detail['range'][1]:.6e}]")
                # 打印最后一层
                last_detail = grad_details[-1]
                print(f"  LAST:  {last_detail['name']}: var={last_detail['var']:.6e}, "
                      f"range=[{last_detail['range'][0]:.6e}, {last_detail['range'][1]:.6e}]")
                # 打印平均值
                print(f"  AVERAGE: {len(grad_variances)} layers, avg_var={raw_variance:.6e}, "
                      f"normalized={normalized_grad_variance:.6e}")
        else:
            print(f"\n=== WARNING: No GCN layer found! ===")
            print("Available weight parameters with gradients:")
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    print(f"  - {name}: shape={param.shape}, "
                          f"grad_range=[{param.grad.min().item():.6e}, {param.grad.max().item():.6e}]")

    optimizer.step()
    
    # 投影权重到列随机/双随机约束（在optimizer.step()之后）
    if project_weights:
        from models_v6 import project_column_stochastic, project_doubly_stochastic, apply_identity_mixing
        threshold = getattr(model, 'w_threshold', w_threshold)
        is_doubly_stochastic = getattr(model, 'doubly_stochastic', False)
        stoch_iters = getattr(model, 'stochastic_iterations', 20)
        
        # EnhancedGCNII: 投影 convs_linear (W')
        if hasattr(model, 'convs_linear'):
            for conv in model.convs_linear:
                if hasattr(conv, 'weight'):
                    if is_doubly_stochastic:
                        project_doubly_stochastic(conv.weight, threshold=threshold, 
                                                 max_iterations=stoch_iters)
                    else:
                        project_column_stochastic(conv.weight, threshold=threshold)
                    if eta > 0 and eta < 1:
                        apply_identity_mixing(conv.weight, eta)
        # NewActivationGNN: 投影 convs (W_l)
        elif hasattr(model, 'convs') and hasattr(model, 'c_activation'):
            for conv in model.convs:
                if hasattr(conv, 'weight'):
                    if is_doubly_stochastic:
                        project_doubly_stochastic(conv.weight, threshold=threshold,
                                                 max_iterations=stoch_iters)
                    else:
                        project_column_stochastic(conv.weight, threshold=threshold)
                    if eta > 0 and eta < 1:
                        apply_identity_mixing(conv.weight, eta)
    
    if track_gradients:
        # 返回loss, acc, (平均梯度方差, 第一层梯度方差, 最后一层梯度方差)
        return loss.item(), acc.item(), (normalized_grad_variance, first_layer_grad_var, last_layer_grad_var)
    else:
        return loss.item(), acc.item()


class PyGModelWrapper(nn.Module):
    """
    PyTorch Geometric模型包装器
    将稠密邻接矩阵格式转换为edge_index格式，用于INT8量化
    """
    def __init__(self, pyg_model):
        super().__init__()
        self.pyg_model = pyg_model
    
    def forward(self, x, adj_dense):
        """
        将稠密邻接矩阵转换为edge_index格式，然后调用PyG模型
        
        Args:
            x: 特征矩阵
            adj_dense: 稠密邻接矩阵
        
        Returns:
            模型输出
        """
        # 从稠密邻接矩阵提取edge_index
        # 找到非零元素的索引
        edge_index = adj_dense.nonzero().t().contiguous()
        
        # 调用PyG模型
        return self.pyg_model(x, edge_index)




def evaluate(model, features, adj, labels, idx, compute_all_metrics=False, onnx_session=None, use_int8_inference=False):
    """
    评估模型（支持PyTorch模型、INT8量化模型或ONNX Runtime）
    
    Args:
        model: PyTorch模型
        features: 特征矩阵
        adj: 邻接矩阵
        labels: 标签
        idx: 评估索引
        compute_all_metrics: 是否计算所有指标
        onnx_session: ONNX Runtime推理会话（已弃用）
        use_int8_inference: 是否使用INT8量化模型
    
    Returns:
        损失和准确率（或所有指标）
    """
    if onnx_session is not None:
        # 使用ONNX Runtime进行推理（已弃用）
        return evaluate_with_onnx(onnx_session, features, adj, labels, idx, compute_all_metrics)
    
    # 检测是否是量化模型（量化模型通常在CPU上）
    is_quantized = use_int8_inference or (hasattr(model, 'qconfig') and model.qconfig is not None)
    
    if is_quantized:
        # INT8量化模型在CPU上运行
        # 将数据移到CPU
        features_cpu = features.cpu()
        labels_cpu = labels.cpu()
        idx_cpu = idx.cpu()
        
        # 处理邻接矩阵
        if adj.is_sparse:
            adj_cpu = adj.cpu().to_dense()
        else:
            adj_cpu = adj.cpu()
        
        # 在CPU上进行推理
        model.eval()
        with torch.no_grad():
            output = model(features_cpu, adj_cpu)
            loss = F.nll_loss(output[idx_cpu], labels_cpu[idx_cpu])
            
            if compute_all_metrics:
                metrics = compute_metrics(output[idx_cpu], labels_cpu[idx_cpu], return_dict=True)
                return loss.item(), metrics
            else:
                pred = output[idx_cpu].max(1)[1]
                acc = pred.eq(labels_cpu[idx_cpu]).sum().item() / idx_cpu.size(0)
                return loss.item(), acc
    else:
        # 使用PyTorch模型进行推理（在GPU上）
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss = F.nll_loss(output[idx], labels[idx])
            if compute_all_metrics:
                metrics = compute_metrics(output[idx], labels[idx], return_dict=True)
                return loss.item(), metrics
            else:
                acc = accuracy(output[idx], labels[idx])
                return loss.item(), acc.item()


def train_and_evaluate(model_type, gamma_value, alpha_value, lambda_value, features, adj, labels,
                       idx_train, idx_val, idx_test, lr, weight_decay, dropout,
                       epochs, patience, nhid, nlayers, device, epsilon=0.1, c_activation=-1.0, 
                       eta=0.0, w_threshold=0.0, verbose=False, fold_idx=0, total_folds=10, model_init_lock=None, 
                       compute_all_metrics=False, activation_type='delta_epsilon', y1=0.05, track_gradients=False,
                       progressive_alpha=False, x1=0.0, x2=1.0, weight_sharing=False, sharing_interval=2,
                       doubly_stochastic=False, stochastic_iterations=20, use_fp8_inference=False, use_int8_inference=False,
                       shared_onnx_path=None, shared_quantized_path=None):
    """训练并评估单个配置"""
    
    # 设置随机种子确保每个配置训练可重复
    # 所有fold使用相同的种子42，确保相同超参数产生相同结果
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    nfeat = features.shape[1]
    nclass = int(labels.max().item() + 1)
    
    # 判断是否是最后一个fold
    is_last_fold = (fold_idx == total_folds - 1)
    
    # 使用锁确保模型初始化是串行的，避免多线程随机状态冲突
    if model_init_lock is not None:
        model_init_lock.acquire()
    
    try:
        # 在创建模型前再次设置种子，确保初始化的确定性
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 创建模型
        if model_type == 'new_activation':
            model = NewActivationGNN(nfeat, nhid, nclass, nlayers,
                                    gamma=gamma_value, alpha=alpha_value, 
                                    epsilon=epsilon, c_activation=c_activation, 
                                    dropout=dropout, verbose=verbose, w_threshold=w_threshold,
                                    activation_type=activation_type, y1=y1, 
                                    progressive_alpha=progressive_alpha, x1=x1, x2=x2,
                                    weight_sharing=weight_sharing, sharing_interval=sharing_interval,
                                    doubly_stochastic=doubly_stochastic, 
                                    stochastic_iterations=stochastic_iterations)
        elif model_type == 'enhanced_gcnii':
            model = EnhancedGCNII(nfeat, nhid, nclass, nlayers,
                                 gamma=gamma_value, alpha=alpha_value, lambda_=lambda_value, dropout=dropout)
        elif model_type == 'standard_gcnii':
            model = StandardGCNII(nfeat, nhid, nclass, nlayers,
                                 alpha=alpha_value, lambda_=lambda_value, dropout=dropout,
                                 weight_sharing=weight_sharing, sharing_interval=sharing_interval)
        elif model_type == 'gcn':
            model = GCN(nfeat, nhid, nclass, nlayers, dropout=dropout)
        elif model_type == 'gat':
            model = GAT(nfeat, nhid, nclass, nlayers, dropout=dropout, heads=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device)
    finally:
        # 释放锁
        if model_init_lock is not None:
            model_init_lock.release()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 初始化训练统计
    training_start_time = time.time()
    gradient_variances = [] if track_gradients else None
    epoch_times = [] if track_gradients else None
    
    # 在最后一个fold开始训练前打印初始化统计
    if is_last_fold and verbose and model_type == 'new_activation':
        model.print_init_stats()
    
    # 训练
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    final_epoch = -1
    
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # EnhancedGCNII和NewActivationGNN都需要投影W到列随机约束
        project_weights = (model_type in ['enhanced_gcnii', 'new_activation'])

        train_result = train_epoch(model, optimizer, features, adj, labels, idx_train, 
                                   project_weights, eta=eta, w_threshold=w_threshold, 
                                   track_gradients=track_gradients, lr=lr, epoch=epoch)
        
        if track_gradients:
            loss_train, acc_train, grad_var_tuple = train_result
            # grad_var_tuple = (avg, first_layer, last_layer)
            if grad_var_tuple is not None and grad_var_tuple[0] is not None:
                gradient_variances.append(grad_var_tuple[0])  # 只保存平均值用于统计
        else:
            loss_train, acc_train = train_result
            
        loss_val, acc_val = evaluate(model, features, adj, labels, idx_val)
        
        epoch_end_time = time.time()
        if track_gradients:
            epoch_times.append(epoch_end_time - epoch_start_time)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        final_epoch = epoch
        if patience_counter >= patience:
            break
    
    # 加载最佳模型并测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 如果是最后一个fold且需要verbose，打印详细统计
    if is_last_fold and verbose and model_type == 'new_activation':
        print(f"\n{'='*80}")
        print(f"最后一个Fold (Fold {fold_idx+1}/{total_folds}) - 最终训练完成后的统计")
        print(f"训练停止于Epoch {final_epoch+1}/{epochs}")
        print(f"{'='*80}\n")
        
        # 启用统计打印并做一次前向传播
        model.enable_print_stats()
        model.train()  # 设置为训练模式以便打印统计
        with torch.no_grad():
            _ = model(features, adj)
        model.disable_print_stats()
    
    # 准备推理阶段的量化
    onnx_session = None
    
    # 如果启用FP8推理，将模型参数转换为FP8
    if use_fp8_inference:
        from models_v6 import convert_model_to_fp8
        if is_last_fold and verbose:
            print(f"\n{'='*80}")
            print(f"将模型参数转换为FP8格式用于推理")
            print(f"{'='*80}\n")
        convert_model_to_fp8(model, use_e4m3=True)
        if is_last_fold and verbose:
            print("模型已转换为FP8格式")
    
    # 如果启用INT8推理，将模型转换为ONNX并量化
    elif use_int8_inference:
        if not INC_AVAILABLE:
            error_msg = "Intel Neural Compressor未安装，无法进行INT8量化。程序终止。"
            print(f"错误: {error_msg}")
            raise RuntimeError(error_msg)
        
        try:
            # 只在第一个fold时进行量化
            if fold_idx == 0:
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"[Fold 0] 使用Intel Neural Compressor进行INT8量化")
                    print(f"{'='*80}\n")
                
                # 准备示例输入用于量化
                # 将稀疏邻接矩阵转换为稠密格式
                if adj.is_sparse:
                    adj_dense = adj.to_dense()
                else:
                    adj_dense = adj
                
                example_inputs = (features, adj_dense)
                
                # 使用INC进行INT8量化
                # 注意：量化后的模型会存储在CPU上
                quantized_model = quantize_model_to_int8(
                    model=model,
                    example_inputs=example_inputs,
                    backend='default'  # 可选: 'fbgemm' (Intel CPU优化) 或 'qnnpack' (ARM)
                )
                
                # 将量化后的模型保存到共享位置（如果提供了）
                if shared_onnx_path is not None:
                    # 使用shared_onnx_path作为模型保存路径
                    model_save_path = shared_onnx_path.replace('.onnx', '_quantized.pth')
                    torch.save(quantized_model.state_dict(), model_save_path)
                    print(f"✓ 量化模型已保存到: {model_save_path}")
                
                # 存储量化模型供后续fold使用
                if not hasattr(train_and_evaluate, '_quantized_models'):
                    train_and_evaluate._quantized_models = {}
                train_and_evaluate._quantized_models[model_type] = quantized_model
                
                if verbose:
                    print(f"✓ INT8量化完成（后续fold将复用此模型）")
            else:
                # 后续fold：复用已量化的模型
                if verbose:
                    print(f"[Fold {fold_idx}] 复用已量化的INT8模型")
                
                if hasattr(train_and_evaluate, '_quantized_models'):
                    quantized_model = train_and_evaluate._quantized_models.get(model_type)
                    if quantized_model is None:
                        raise RuntimeError("未找到已量化的模型，这不应该发生")
                else:
                    raise RuntimeError("未找到已量化的模型，这不应该发生")
            
            # 用量化模型替换原始模型用于推理
            # 注意：quantized_model在CPU上
            model = quantized_model
            model.eval()
            
        except Exception as e:
            error_msg = f"INT8量化失败: {e}\n量化失败，程序终止。"
            print(f"错误: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    loss_test, result = evaluate(model, features, adj, labels, idx_test, 
                                 compute_all_metrics=compute_all_metrics,
                                 onnx_session=onnx_session,
                                 use_int8_inference=use_int8_inference)
    
    # 计算总训练时间
    training_time = time.time() - training_start_time

    # 如果需要追踪梯度，也计算信号增益
    signal_gains = []
    if track_gradients:
        # 提取所有卷积层的权重矩阵
        weights = []
        if model_type == 'new_activation':
            # NewActivationGNN: 权重在 convs 中的 weight 属性
            for conv_layer in model.convs:
                if hasattr(conv_layer, 'weight'):
                    weights.append(conv_layer.weight.detach().cpu().numpy())
        elif model_type == 'gcn':
            # GCN: 使用 PyTorch Geometric 的 GCNConv，权重在 lin.weight
            for conv_layer in model.convs:
                if hasattr(conv_layer, 'lin') and hasattr(conv_layer.lin, 'weight'):
                    weights.append(conv_layer.lin.weight.detach().cpu().numpy())
                elif hasattr(conv_layer, 'weight'):
                    weights.append(conv_layer.weight.detach().cpu().numpy())
        elif model_type == 'gat':
            # GAT: 使用 PyTorch Geometric 的 GATConv，权重在 lin.weight
            for conv_layer in model.convs:
                if hasattr(conv_layer, 'lin') and hasattr(conv_layer.lin, 'weight'):
                    weights.append(conv_layer.lin.weight.detach().cpu().numpy())
                elif hasattr(conv_layer, 'lin_src'):
                    weights.append(conv_layer.lin_src.weight.detach().cpu().numpy())
        
        # 计算每层的信号增益
        if len(weights) > 0:
            L = len(weights)
            for l in range(L):
                # 计算 W_{L-1} * W_{L-2} * ... * W_l
                product = weights[L-1].copy()
                valid_product = True
                for i in range(L-2, l-1, -1):
                    try:
                        product = np.dot(product, weights[i])
                    except ValueError as e:
                        # 如果矩阵维度不匹配，标记为无效
                        print(f"[Signal Gain] Matrix multiplication failed at layer {i}: {e}")
                        valid_product = False
                        break
                
                if valid_product:
                    # 取绝对值，按行求和，取最大值
                    abs_product = np.abs(product)
                    row_sums = np.sum(abs_product, axis=1)
                    max_gain = np.max(row_sums)
                    signal_gains.append(max_gain)
                else:
                    # 如果计算失败，添加 0
                    signal_gains.append(0.0)
            
            print(f"[Signal Gain Debug] Model: {model_type}, Extracted {len(weights)} weight matrices, Computed {len(signal_gains)} signal gains")

    # 清理模型和优化器以释放GPU内存
    del model, optimizer, best_model_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    if track_gradients:
        training_stats = {
            'training_time': training_time,
            'gradient_variances': gradient_variances if gradient_variances else [],
            'epoch_times': epoch_times if epoch_times else [],
            'final_epoch': final_epoch + 1,
            'signal_gains': signal_gains
        }
        if compute_all_metrics:
            return result, training_stats  # 返回字典和统计信息
        else:
            return result, training_stats  # 返回 acc_test 和统计信息
    else:
        if compute_all_metrics:
            return result  # 返回字典
        else:
            return result  # 返回 acc_test


def k_fold_cv(model_type, gamma_value, alpha_value, lambda_value, features, adj, labels, indices,
              lr, weight_decay, dropout, epochs, patience, nhid, nlayers, device, k=10, epsilon=0.1, 
              c_activation=-1.0, eta=0.0, w_threshold=0.0, verbose=False, model_init_lock=None, 
              compute_all_metrics=False, activation_type='delta_epsilon', y1=0.05, track_gradients=False,
              progressive_alpha=False, x1=0.0, x2=1.0, weight_sharing=False, sharing_interval=2,
              doubly_stochastic=False, stochastic_iterations=20, use_fp8_inference=False, use_int8_inference=False):
    """
    10-fold交叉验证（分层抽样）
    确保训练集、验证集和测试集中每个类别的比例大致相同
    """
    
    # 如果启用INT8推理，创建一个共享的临时目录和文件路径
    shared_onnx_path = None
    shared_quantized_path = None
    shared_temp_dir = None
    if use_int8_inference:
        import tempfile
        shared_temp_dir = tempfile.mkdtemp()
        shared_onnx_path = os.path.join(shared_temp_dir, 'model.onnx')
        shared_quantized_path = os.path.join(shared_temp_dir, 'model_int8.onnx')
    
    # 获取标签用于分层
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 检查是否所有类别都有至少k个样本（用于分层K-fold）
    unique_labels, label_counts = np.unique(labels_np[indices], return_counts=True)
    min_samples_per_class = label_counts.min()
    use_stratified = min_samples_per_class >= k
    
    if not use_stratified:
        print(f"警告: 某些类别样本数少于 {k}（最小={min_samples_per_class}），使用普通K-fold而非分层K-fold")
        skfold = KFold(n_splits=k, shuffle=True, random_state=42)
    else:
        # 使用分层 K-fold 交叉验证，确保每个fold中类别分布相同
        skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skfold.split(indices, labels_np[indices] if use_stratified else None)):
        # 第一步：划分为 train_val (90%) 和 test (10%)
        train_val_indices = indices[train_val_idx]
        test_indices = indices[test_idx]
        
        # 第二步：在 train_val 中划分为 train (80%) 和 val (10%)
        # 检查train_val中每个类别是否至少有2个样本（用于分层划分）
        train_val_labels = labels_np[train_val_indices]
        _, train_val_counts = np.unique(train_val_labels, return_counts=True)
        use_stratified_split = train_val_counts.min() >= 2
        
        if use_stratified_split:
            # 使用 train_test_split 进行分层抽样
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=1/9,  # 10% / 90% = 1/9，使得 val 占总数据的 10%
                stratify=train_val_labels,  # 按标签分层
                random_state=42 + fold  # 每个fold使用不同的随机种子
            )
        else:
            # 使用普通随机抽样
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=1/9,  # 10% / 90% = 1/9，使得 val 占总数据的 10%
                random_state=42 + fold  # 每个fold使用不同的随机种子
            )
        
        idx_train = torch.LongTensor(train_indices).to(device)
        idx_val = torch.LongTensor(val_indices).to(device)
        idx_test = torch.LongTensor(test_indices).to(device)
        
        # 训练并评估（只在最后一个fold的最后一个epoch输出详细信息）
        result = train_and_evaluate(model_type, gamma_value, alpha_value, lambda_value,
                                   features, adj, labels, idx_train, idx_val, idx_test,
                                   lr, weight_decay, dropout, epochs, patience,
                                   nhid, nlayers, device, epsilon, c_activation,
                                   eta=eta, w_threshold=w_threshold, verbose=verbose, fold_idx=fold, total_folds=k,
                                   model_init_lock=model_init_lock, compute_all_metrics=compute_all_metrics,
                                   activation_type=activation_type, y1=y1, track_gradients=track_gradients,
                                   progressive_alpha=progressive_alpha, x1=x1, x2=x2,
                                   weight_sharing=weight_sharing, sharing_interval=sharing_interval,
                                   doubly_stochastic=doubly_stochastic, stochastic_iterations=stochastic_iterations,
                                   use_fp8_inference=use_fp8_inference, use_int8_inference=use_int8_inference,
                                   shared_onnx_path=shared_onnx_path, shared_quantized_path=shared_quantized_path)

        # 处理返回值：如果track_gradients=True，返回值是(result, stats)的元组
        if track_gradients:
            metrics, stats = result
            fold_results.append(metrics)
            # 对于对比实验，我们只需要第一个fold的统计信息
            if fold == 0:
                fold_stats = stats
        else:
            fold_results.append(result)
    
    # 清理共享的临时目录
    if use_int8_inference and shared_temp_dir is not None:
        try:
            import shutil
            shutil.rmtree(shared_temp_dir)
        except:
            pass
    
    if compute_all_metrics:
        # 返回所有指标的均值和标准差
        avg_metrics = {
            'accuracy': (np.mean([r['accuracy'] for r in fold_results]), 
                       np.std([r['accuracy'] for r in fold_results])),
            'macro_f1': (np.mean([r['macro_f1'] for r in fold_results]), 
                       np.std([r['macro_f1'] for r in fold_results])),
            'macro_precision': (np.mean([r['macro_precision'] for r in fold_results]), 
                              np.std([r['macro_precision'] for r in fold_results])),
            'macro_recall': (np.mean([r['macro_recall'] for r in fold_results]), 
                           np.std([r['macro_recall'] for r in fold_results]))
        }
        if track_gradients:
            return avg_metrics, fold_stats
        else:
            return avg_metrics
    else:
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        if track_gradients:
            return (mean_acc, std_acc), fold_stats
        else:
            return mean_acc, std_acc


def train_single_baseline_model(model_name, model_class, features, edge_index, labels, indices,
                               lr, weight_decay, dropout, epochs, patience, nhid, nlayers,
                               device, k=10, compute_all_metrics=True, print_lock=None, alpha=0.5,
                               weight_sharing=False, sharing_interval=2, use_int8_inference=False):
    """训练单个基线模型（用于并行）- 优化内存使用"""
    
    # 内存优化：在函数内部按需拷贝数据，避免多个GPU同时保留完整数据副本
    features_gpu = features.to(device)
    edge_index_gpu = edge_index.to(device)
    labels_gpu = labels.to(device)

    nfeat = features_gpu.shape[1]
    nclass = int(labels_gpu.max().item() + 1)

    # 检查是否所有类别都有至少k个样本（用于分层K-fold）
    labels_np = labels_gpu.cpu().numpy() if isinstance(labels_gpu, torch.Tensor) else labels_gpu
    unique_labels, label_counts = np.unique(labels_np[indices], return_counts=True)
    min_samples_per_class = label_counts.min()
    use_stratified = min_samples_per_class >= k

    if not use_stratified:
        with print_lock if print_lock else threading.Lock():
            print(f"[GPU {device.index if device.type == 'cuda' else 'CPU'}] {model_name} 警告: 某些类别样本数少于 {k}（最小={min_samples_per_class}），使用普通K-fold而非分层K-fold")
        skfold = KFold(n_splits=k, shuffle=True, random_state=42)
    else:
        # 使用分层 K-fold 交叉验证，确保每个fold中类别分布相同
        skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    with print_lock if print_lock else threading.Lock():
        print(f"\n{'='*80}")
        print(f"[GPU {device.index if device.type == 'cuda' else 'CPU'}] 训练基线模型: {model_name}")
        print(f"{'='*80}")
        
    fold_results = []
        
    for fold, (train_val_idx, test_idx) in enumerate(skfold.split(indices, labels_np[indices] if use_stratified else None)):
        # 每个fold开始前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        # 划分数据集
        train_val_indices = indices[train_val_idx]
        test_indices = indices[test_idx]
            
        # 检查train_val中每个类别是否至少有2个样本（用于分层划分）
        train_val_labels = labels_np[train_val_indices]
        _, train_val_counts = np.unique(train_val_labels, return_counts=True)
        use_stratified_split = train_val_counts.min() >= 2

        if use_stratified_split:
            # 使用 train_test_split 进行分层抽样
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=1/9,  # 10% / 90% = 1/9，使得 val 占总数据的 10%
                stratify=train_val_labels,  # 按标签分层
                random_state=42 + fold  # 每个fold使用不同的随机种子
            )
        else:
            # 使用普通随机抽样
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=1/9,  # 10% / 90% = 1/9，使得 val 占总数据的 10%
                random_state=42 + fold  # 每个fold使用不同的随机种子
            )
        
        # 设置随机种子
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 创建模型
        if model_name == 'GCN':
            model = model_class(nfeat, nhid, nclass, nlayers, dropout=dropout,
                              weight_sharing=weight_sharing, sharing_interval=sharing_interval)
        elif model_name == 'GAT':
            model = model_class(nfeat, nhid, nclass, nlayers, dropout=dropout, heads=1,
                              weight_sharing=weight_sharing, sharing_interval=sharing_interval)
        elif model_name == 'APPNP':
            model = model_class(nfeat, nhid, nclass, nlayers, dropout=dropout, alpha=0.9, K=5)
        elif model_name == 'SGC':
            model = model_class(nfeat, nhid, nclass, nlayers, dropout=dropout, K=20)
        elif model_name == 'GCNII':
            model = model_class(nfeat, nhid, nclass, nlayers, alpha=alpha, dropout=dropout)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 训练前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # 训练
        best_val_metric = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            model.train()
            optimizer.zero_grad()
            
            # GCNII使用邻接矩阵格式，其他模型使用edge_index格式
            if model_name == 'GCNII':
                # 将edge_index转换为稀疏邻接矩阵
                num_nodes = features_gpu.size(0)
                adj_sparse = torch.sparse_coo_tensor(
                    edge_index_gpu, 
                    torch.ones(edge_index_gpu.size(1), device=device),
                    size=(num_nodes, num_nodes)
                )
                output = model(features_gpu, adj_sparse)
            else:
                output = model(features_gpu, edge_index_gpu)
            
            loss = F.nll_loss(output[train_indices], labels_gpu[train_indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                # GCNII使用邻接矩阵格式，其他模型使用edge_index格式
                if model_name == 'GCNII':
                    num_nodes = features_gpu.size(0)
                    adj_sparse = torch.sparse_coo_tensor(
                        edge_index_gpu, 
                        torch.ones(edge_index_gpu.size(1), device=device),
                        size=(num_nodes, num_nodes)
                    )
                    output = model(features_gpu, adj_sparse)
                else:
                    output = model(features_gpu, edge_index_gpu)
                val_acc = accuracy(output[val_indices], labels_gpu[val_indices])
            
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # 测试
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 如果启用INT8推理，使用INC进行量化
        quantized_model_for_inference = None
        if use_int8_inference and INC_AVAILABLE:
            try:
                if print_lock:
                    with print_lock:
                        print(f"  [{model_name}] 使用Intel Neural Compressor进行INT8量化...")
                else:
                    print(f"  [{model_name}] 使用Intel Neural Compressor进行INT8量化...")
                
                # 准备示例输入用于量化
                # 构建稠密邻接矩阵
                num_nodes = features_gpu.size(0)
                adj_dense = torch.zeros(num_nodes, num_nodes, device=device)
                adj_dense[edge_index_gpu[0], edge_index_gpu[1]] = 1.0
                
                # 如果是PyG模型，需要使用包装器
                if model_name != 'GCNII':
                    # PyG模型（GCN, GAT等）需要包装成接受稠密邻接矩阵的格式
                    wrapped_model = PyGModelWrapper(model)
                    example_inputs = (features_gpu, adj_dense)
                    quantized_model_for_inference = quantize_model_to_int8(
                        model=wrapped_model,
                        example_inputs=example_inputs,
                        backend='default'
                    )
                else:
                    # GCNII直接使用稠密邻接矩阵
                    example_inputs = (features_gpu, adj_dense)
                    quantized_model_for_inference = quantize_model_to_int8(
                        model=model,
                        example_inputs=example_inputs,
                        backend='default'
                    )
                
                # 确保量化模型在CPU上（INT8推理在CPU上进行）
                if quantized_model_for_inference is None:
                    error_msg = f"{model_name} INT8量化返回None"
                    if print_lock:
                        with print_lock:
                            print(f"错误: {error_msg}")
                    else:
                        print(f"错误: {error_msg}")
                    raise RuntimeError(error_msg)
                
                quantized_model_for_inference = quantized_model_for_inference.cpu()
                quantized_model_for_inference.eval()
                
                if print_lock:
                    with print_lock:
                        print(f"  [{model_name}] ✓ INT8量化成功")
                else:
                    print(f"  [{model_name}] ✓ INT8量化成功")
                
            except Exception as e:
                error_msg = f"{model_name} INT8量化失败: {e}\n量化失败，程序终止。"
                if print_lock:
                    with print_lock:
                        print(f"错误: {error_msg}")
                else:
                    print(f"错误: {error_msg}")
                raise RuntimeError(error_msg) from e
        
        model.eval()
        
        # 使用量化模型或标准PyTorch进行推理
        if quantized_model_for_inference is not None:
            # INT8量化推理（在CPU上）
            # 将数据移到CPU
            features_cpu = features_gpu.cpu()
            num_nodes = features_cpu.size(0)
            adj_dense_cpu = torch.zeros(num_nodes, num_nodes)
            adj_dense_cpu[edge_index_gpu[0].cpu(), edge_index_gpu[1].cpu()] = 1.0
            
            with torch.no_grad():
                output_cpu = quantized_model_for_inference(features_cpu, adj_dense_cpu)
                # 将输出移回GPU用于指标计算
                output = output_cpu.to(device)
        else:
            # 标准PyTorch推理
            with torch.no_grad():
                # GCNII使用邻接矩阵格式，其他模型使用edge_index格式
                if model_name == 'GCNII':
                    num_nodes = features_gpu.size(0)
                    adj_sparse = torch.sparse_coo_tensor(
                        edge_index_gpu, 
                        torch.ones(edge_index_gpu.size(1), device=device),
                        size=(num_nodes, num_nodes)
                    )
                    output = model(features_gpu, adj_sparse)
                else:
                    output = model(features_gpu, edge_index_gpu)
        
        if compute_all_metrics:
            metrics = compute_metrics(output[test_indices], labels_gpu[test_indices], return_dict=True)
            fold_results.append(metrics)
        else:
            acc = accuracy(output[test_indices], labels_gpu[test_indices]).item()
            fold_results.append({'accuracy': acc})
            
        with print_lock if print_lock else threading.Lock():
            print(f"  [GPU {device.index if device.type == 'cuda' else 'CPU'}] {model_name} Fold {fold+1}/{k}: Acc={fold_results[-1]['accuracy']:.4f}")

        # 清理内存 - 更激进的清理
        del model, optimizer
        if 'best_model_state' in locals() and best_model_state is not None:
            del best_model_state
        if 'quantized_model_for_inference' in locals() and quantized_model_for_inference is not None:
            del quantized_model_for_inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 强制同步以确保清理完成
            torch.cuda.synchronize()
        import gc
        gc.collect()
        
        # 计算平均值和标准差
        if compute_all_metrics:
            avg_metrics = {
                'accuracy': (np.mean([r['accuracy'] for r in fold_results]), 
                           np.std([r['accuracy'] for r in fold_results])),
                'macro_f1': (np.mean([r['macro_f1'] for r in fold_results]), 
                           np.std([r['macro_f1'] for r in fold_results])),
                'macro_precision': (np.mean([r['macro_precision'] for r in fold_results]), 
                                  np.std([r['macro_precision'] for r in fold_results])),
                'macro_recall': (np.mean([r['macro_recall'] for r in fold_results]), 
                               np.std([r['macro_recall'] for r in fold_results]))
            }
        else:
            avg_metrics = {
                'accuracy': (np.mean([r['accuracy'] for r in fold_results]), 
                           np.std([r['accuracy'] for r in fold_results]))
            }
        
    with print_lock if print_lock else threading.Lock():
        print(f"\n[GPU {device.index if device.type == 'cuda' else 'CPU'}] {model_name} 平均准确率: {avg_metrics['accuracy'][0]:.4f} ± {avg_metrics['accuracy'][1]:.4f}")

    # 清理GPU数据副本
    del features_gpu, edge_index_gpu, labels_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return model_name, avg_metrics


def train_baseline_models(features, adj, labels, indices, lr, weight_decay, dropout, epochs,
                         patience, nhid, nlayers, device, k=10, compute_all_metrics=True,
                         baseline_models_to_run=None, alpha=0.5, use_int8_inference=False):
    """训练多个基线模型并返回结果（单GPU顺序运行）"""

    # 将邻接矩阵转换为edge_index格式（PyG格式）
    edge_index = adj._indices()

    # 所有可用的基线模型
    all_baseline_models = {
        'GCN': GCN,
        'GAT': GAT,
        'APPNP': APPNP,
        'SGC': SGC,
        'GCNII': StandardGCNII
    }

    # 确定要运行的模型
    if baseline_models_to_run is None or len(baseline_models_to_run) == 0:
        # 运行所有模型（包括GCNII）
        baseline_models_to_run = list(all_baseline_models.keys())
    else:
        # 检查指定的模型是否都支持
        for model_name in baseline_models_to_run:
            if model_name not in all_baseline_models:
                raise ValueError(f"不支持的基线模型: {model_name}")

    baseline_results = {}

    # 按顺序训练每个基线模型
    for baseline_model_name in baseline_models_to_run:
        # 每个模型开始前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

        model_class = all_baseline_models[baseline_model_name]

        print(f"\n{'-'*60}")
        print(f"训练基线模型: {baseline_model_name}")
        print(f"{'-'*60}")

        # 训练单个基线模型
        model_name, avg_metrics = train_single_baseline_model(
            baseline_model_name, model_class, features, edge_index, labels, indices,
            lr, weight_decay, dropout, epochs, patience, nhid, nlayers,
            device, k, compute_all_metrics, alpha=alpha,
            use_int8_inference=use_int8_inference
        )

        baseline_results[model_name] = avg_metrics

        # 每个基线模型训练完成后清理GPU内存
        # 注意：train_single_baseline_model函数内部已经清理了GPU数据副本
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"模型 {baseline_model_name} 显存已清理\n")

    # 清理edge_index
    del edge_index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return baseline_results


def run_weight_sharing_experiment(features, adj, labels, indices, args, device):
    """
    权重共享实验：每y层共享权重矩阵W
    
    对 GCN、GAT、GCNII 和优化模型进行实验
    """
    print("\n" + "="*80)
    print("WEIGHT SHARING EXPERIMENT")
    print("="*80)
    print(f"权重共享间隔: 每 {args.sharing_interval} 层共享权重")
    print(f"总层数: {args.layers}")
    print(f"数据集: {args.dataset}")
    print(f"设备: {device}")
    print("="*80 + "\n")
    
    # 将数据移动到指定设备（非常重要！）
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    
    models_to_test = ['GCN', 'GAT', 'GCNII', 'NewActivationGNN']
    results = {}
    
    # 将邻接矩阵转换为edge_index格式（PyG格式）
    edge_index = adj._indices()
    
    for model_name in models_to_test:
        print(f"\n{'-'*60}")
        print(f"测试模型: {model_name} (权重共享间隔={args.sharing_interval})")
        print(f"{'-'*60}")
        
        try:
            if model_name == 'NewActivationGNN':
                # 优化模型
                result = k_fold_cv('new_activation', args.gamma_values[0], 0.9, 0.5,
                                  features, adj, labels, indices,
                                  args.lr, args.weight_decay, args.dropout,
                                  args.epochs, args.patience, args.hidden,
                                  args.layers, device, k=args.k_folds,
                                  epsilon=args.epsilon,
                                  c_activation=args.c_activation[0] if isinstance(args.c_activation, list) else args.c_activation,
                                  eta=args.eta, w_threshold=args.w_threshold,
                                  verbose=False, model_init_lock=None,
                                  compute_all_metrics=False,
                                  activation_type=args.activation_type, y1=args.y1,
                                  use_fp8_inference=args.use_fp8_inference,
                                  use_int8_inference=args.use_int8_inference,
                                  track_gradients=False, progressive_alpha=args.progressive_alpha,
                                  x1=args.x1, x2=args.x2,
                                  weight_sharing=True, sharing_interval=args.sharing_interval,
                                  doubly_stochastic=args.doubly_stochastic,
                                  stochastic_iterations=args.stochastic_iterations)
                mean_acc, std_acc = result
            elif model_name == 'GCNII':
                # GCNII基线
                result = k_fold_cv('standard_gcnii', 0, 0.9, 0.5,
                                  features, adj, labels, indices,
                                  args.lr, args.weight_decay, args.dropout,
                                  args.epochs, args.patience, args.hidden,
                                  args.layers, device, k=args.k_folds,
                                  epsilon=0, c_activation=-1.0, eta=0.0, w_threshold=0.0,
                                  verbose=False, model_init_lock=None,
                                  compute_all_metrics=False,
                                  track_gradients=False,
                                  weight_sharing=True, sharing_interval=args.sharing_interval,
                                  use_fp8_inference=args.use_fp8_inference,
                                  use_int8_inference=args.use_int8_inference)
                mean_acc, std_acc = result
            else:
                # GCN, GAT基线
                model_class_map = {
                    'GCN': GCN,
                    'GAT': GAT
                }
                model_class = model_class_map[model_name]
                _, avg_metrics = train_single_baseline_model(
                    model_name, model_class, features, edge_index, labels, indices,
                    args.lr, args.weight_decay, args.dropout,
                    args.epochs, args.patience, args.hidden, args.layers,
                    device, k=args.k_folds, compute_all_metrics=False,
                    print_lock=threading.Lock(), alpha=0.9,
                    weight_sharing=True, sharing_interval=args.sharing_interval,
                    use_int8_inference=args.use_int8_inference
                )
                mean_acc, std_acc = avg_metrics['accuracy']
            
            results[model_name] = {'accuracy': (mean_acc, std_acc)}
            print(f"{model_name} 准确率: {mean_acc:.4f} ± {std_acc:.4f}")
            
        except Exception as e:
            print(f"{model_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'accuracy': (0.0, 0.0)}
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 清理edge_index
    del edge_index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 输出结果对比
    print("\n" + "="*80)
    print("权重共享实验结果对比")
    print("="*80)
    
    opt_acc, opt_std = results['NewActivationGNN']['accuracy']
    
    for model_name in ['GCN', 'GAT', 'GCNII']:
        if model_name in results:
            baseline_acc, baseline_std = results[model_name]['accuracy']
            if baseline_acc > 0:
                improvement = ((opt_acc - baseline_acc) / baseline_acc * 100)
                print(f"\n{model_name}:")
                print(f"  准确率: {baseline_acc:.4f} ± {baseline_std:.4f}")
                print(f"  优化模型提升: {improvement:+.2f}%")
    
    print(f"\nNewActivationGNN:")
    print(f"  准确率: {opt_acc:.4f} ± {opt_std:.4f}")
    print("="*80 + "\n")
    
    return results


def benchmark_time_across_datasets(args, device):
    """
    时间基准测试：比较多个数据集在多个模型上的10折交叉验证总时间
    
    比较指标：
    - 10-Fold Cross-Validation Total Time: 完整10折交叉验证的总时间（最权威的比较指标）
    
    数据集：'texas', 'cora', 'chameleon', 'citeseer', 'squirrel', 'pubmed'
    模型：GCN, GAT, APPNP, SGC, GCNII, NewActivationGNN (优化模型)
    
    输出：
    - 表格（横轴：模型，纵轴：数据集）
    - 最后一列：Speedup（优化模型相比最慢模型的加速比）
    """
    datasets = ['texas', 'cora', 'chameleon', 'citeseer', 'squirrel', 'pubmed']
    models = ['GCN', 'GAT', 'APPNP', 'SGC', 'GCNII', 'NewActivationGNN']
    
    print("\n" + "="*100)
    print("TIME BENCHMARK: 10-Fold Cross-Validation Total Time Comparison")
    print("="*100)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(models)}")
    print(f"Metric: Total time for complete 10-fold cross-validation (seconds)")
    print("="*100 + "\n")
    
    # 存储结果: results[dataset][model] = time_in_seconds
    results = {dataset: {} for dataset in datasets}
    
    seed = 42
    for dataset_name in datasets:
        print(f"\n{'='*100}")
        print(f"Processing Dataset: {dataset_name.upper()}")
        print(f"{'='*100}\n")
        
        # 加载数据
        try:
            adj, features, labels, _, _, _ = load_data(dataset_name, use_standard_split=True, seed=seed)
            num_nodes = features.shape[0]
            all_indices = np.arange(num_nodes)
            nfeat = features.shape[1]
            nclass = int(labels.max().item() + 1)
            
            print(f"Dataset: {dataset_name}")
            print(f"  Nodes: {num_nodes}, Features: {nfeat}, Classes: {nclass}")
            print(f"  Device: {device}\n")
            
            # 将数据移到GPU
            features_gpu = features.to(device)
            adj_gpu = adj.to(device)
            labels_gpu = labels.to(device)
            
            # 测试每个模型
            for model_name in models:
                print(f"  Testing {model_name}...")
                
                try:
                    start_time = time.time()
                    
                    if model_name == 'NewActivationGNN':
                        # 优化模型：使用提供的超参数
                        _ = k_fold_cv('new_activation', args.gamma_values[0], 0.9, 0.5,
                                     features_gpu, adj_gpu, labels_gpu, all_indices,
                                     args.lr, args.weight_decay, args.dropout,
                                     args.epochs, args.patience, args.hidden,
                                     args.layers, device, k=10,
                                     epsilon=args.epsilon, 
                                     c_activation=args.c_activation[0] if isinstance(args.c_activation, list) else args.c_activation,
                                     eta=args.eta, w_threshold=args.w_threshold, 
                                     verbose=False, model_init_lock=None,
                                     compute_all_metrics=False,
                                     activation_type=args.activation_type, y1=args.y1,
                                     track_gradients=False, progressive_alpha=args.progressive_alpha,
                                     x1=args.x1, x2=args.x2,
                                     doubly_stochastic=args.doubly_stochastic,
                                     stochastic_iterations=args.stochastic_iterations,
                                     use_fp8_inference=args.use_fp8_inference,
                                     use_int8_inference=args.use_int8_inference)
                    elif model_name == 'GCNII':
                        # GCNII：使用标准配置
                        _ = k_fold_cv('standard_gcnii', 0, 0.9, 0.5,
                                     features_gpu, adj_gpu, labels_gpu, all_indices,
                                     args.lr, args.weight_decay, args.dropout,
                                     args.epochs, args.patience, args.hidden,
                                     args.layers, device, k=10,
                                     epsilon=0, c_activation=-1.0, eta=0.0, w_threshold=0.0,
                                     verbose=False, model_init_lock=None,
                                     compute_all_metrics=False,
                                     track_gradients=False,
                                     use_fp8_inference=args.use_fp8_inference,
                                     use_int8_inference=args.use_int8_inference)
                    else:
                        # 其他基线模型
                        model_class_map = {
                            'GCN': GCN,
                            'GAT': GAT,
                            'APPNP': APPNP,
                            'SGC': SGC
                        }
                        
                        # 转换adj为edge_index
                        from torch_geometric.utils import from_scipy_sparse_matrix
                        if adj_gpu.is_sparse:
                            adj_coo = adj_gpu.coalesce()
                            edge_index = torch.stack([adj_coo.indices()[0], adj_coo.indices()[1]], dim=0)
                        else:
                            # 如果是稠密矩阵，转换为COO格式
                            import scipy.sparse as sp
                            adj_np = adj_gpu.cpu().numpy()
                            adj_sp = sp.coo_matrix(adj_np)
                            edge_index, _ = from_scipy_sparse_matrix(adj_sp)
                            edge_index = edge_index.to(device)
                        
                        # 运行基线模型
                        baseline_result = train_single_baseline_model(
                            model_name, model_class_map[model_name],
                            features_gpu, edge_index, labels_gpu, all_indices,
                            args.lr, args.weight_decay, args.dropout,
                            args.epochs, args.patience, args.hidden, args.layers,
                            device, k=10, compute_all_metrics=False,
                            print_lock=threading.Lock(), alpha=0.9,
                            use_int8_inference=args.use_int8_inference
                        )
                    
                    elapsed_time = time.time() - start_time
                    results[dataset_name][model_name] = elapsed_time
                    print(f"    ✓ {model_name}: {elapsed_time:.2f}s")
                    
                except Exception as e:
                    print(f"    ✗ {model_name} failed: {e}")
                    results[dataset_name][model_name] = -1  # 标记失败
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # 清理数据集
            del features_gpu, adj_gpu, labels_gpu, features, adj, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Failed to load dataset {dataset_name}: {e}")
            for model_name in models:
                results[dataset_name][model_name] = -1
    
    # 输出结果表格
    print("\n" + "="*100)
    print("BENCHMARK RESULTS: 10-Fold Cross-Validation Total Time (seconds)")
    print("="*100 + "\n")
    
    # 表头
    header = f"{'Dataset':<15}"
    for model in models:
        header += f"{model:>12}"
    header += f"{'Max Speedup':>15}"  # 优化模型相比最慢模型的加速比
    print(header)
    print("-" * 100)
    
    # 数据行
    for dataset_name in datasets:
        row = f"{dataset_name:<15}"
        times = []
        for model in models:
            t = results[dataset_name].get(model, -1)
            if t > 0:
                row += f"{t:>12.2f}"
                times.append((model, t))
            else:
                row += f"{'FAIL':>12}"
        
        # 计算Speedup
        if len(times) > 0:
            opt_time = results[dataset_name].get('NewActivationGNN', -1)
            if opt_time > 0:
                # 找到最慢的模型
                slowest_time = max(t for _, t in times)
                speedup = slowest_time / opt_time
                row += f"{speedup:>15.2f}x"
            else:
                row += f"{'N/A':>15}"
        else:
            row += f"{'N/A':>15}"
        
        print(row)
    
    print("\n" + "="*100)
    print("Note:")
    print("  - Time: Total time for complete 10-fold cross-validation")
    print("  - Max Speedup: NewActivationGNN time / Slowest baseline time")
    print("  - Higher speedup means the optimized model is faster")
    print("="*100 + "\n")
    
    return results


def compare_models_training(features, adj, labels, indices, lr, weight_decay, dropout,
                          epochs, patience, nhid, nlayers, device, epsilon, eta, w_threshold,
                          best_gamma, best_alpha, best_lambda, best_c_activation, 
                          activation_type='delta_epsilon', y1=0.05, progressive_alpha=False, x1=0.0, x2=1.0,
                          doubly_stochastic=False, stochastic_iterations=20):
    """
    比较优化模型、GCN和GAT的训练效率

    指标：
    1. 时间指标：
       - Average Time per Epoch: 每个epoch的平均训练时间
       - Total Training Time: 单个fold完整训练的总时间

    2. 梯度指标（所有GCN层权重矩阵的平均）：
       - Normalized Gradient Variance: Var[∂L/∂W] / (lr^2)
         这是学习率归一化的梯度方差，消除学习率对梯度方差的影响
    """
    print("\n" + "="*80)
    print("Training Efficiency Comparison: Optimized Model vs GCN vs GAT")
    print("="*80)

    # 确保数据在正确的设备上
    features_gpu = features.to(device)
    adj_gpu = adj.to(device)
    labels_gpu = labels.to(device)

    results = {}
    
    # 使用单个fold进行对比（减少计算时间，同时保持公平性）
    # 准备数据分割
    from sklearn.model_selection import StratifiedKFold, train_test_split
    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    labels_np = labels_gpu.cpu().numpy()
    
    fold_idx = 0
    for train_val_idx, test_idx in skfold.split(indices, labels_np[indices]):
        if fold_idx == 0:  # 只使用第一个fold
            train_val_indices = indices[train_val_idx]
            test_indices = indices[test_idx]
            
            # 进一步分割为train和val
            train_val_labels = labels_np[train_val_indices]
            _, train_val_counts = np.unique(train_val_labels, return_counts=True)
            use_stratified_split = train_val_counts.min() >= 2
            
            if use_stratified_split:
                train_indices, val_indices = train_test_split(
                    train_val_indices, test_size=1/9, stratify=train_val_labels, random_state=42)
            else:
                train_indices, val_indices = train_test_split(
                    train_val_indices, test_size=1/9, random_state=42)
            
            idx_train = torch.LongTensor(train_indices).to(device)
            idx_val = torch.LongTensor(val_indices).to(device)
            idx_test = torch.LongTensor(test_indices).to(device)
            break
        fold_idx += 1

    # 测试优化模型
    print("\n--- Testing Optimized Model ---")
    opt_result, opt_stats = train_and_evaluate(
        'new_activation', best_gamma, best_alpha, 0.5,
        features_gpu, adj_gpu, labels_gpu, idx_train, idx_val, idx_test,
        lr, weight_decay, dropout, epochs, patience, nhid, nlayers, device,
        epsilon=epsilon, c_activation=best_c_activation, eta=eta, w_threshold=w_threshold,
        verbose=False, fold_idx=0, total_folds=1, model_init_lock=None,
        compute_all_metrics=False, activation_type=activation_type, y1=y1, 
        track_gradients=True, progressive_alpha=progressive_alpha, x1=x1, x2=x2,
        doubly_stochastic=doubly_stochastic, stochastic_iterations=stochastic_iterations,
        use_fp8_inference=False
    )
    
    results['optimized'] = {
        'test_acc': opt_result,
        'training_time': opt_stats['training_time'],
        'gradient_variances': opt_stats['gradient_variances'],
        'epoch_times': opt_stats['epoch_times'],
        'num_epochs': opt_stats['final_epoch'],
        'signal_gains': opt_stats.get('signal_gains', [])
    }

    # 测试GCN
    print("--- Testing GCN ---")
    try:
        gcn_result, gcn_stats = train_and_evaluate(
            'gcn', 0, 0, 0,  # GCN不需要这些参数
            features_gpu, adj_gpu, labels_gpu, idx_train, idx_val, idx_test,
            lr, weight_decay, dropout, epochs, patience, nhid, nlayers, device,
            epsilon=0, c_activation=0, eta=0, w_threshold=0,
            verbose=False, fold_idx=0, total_folds=1, model_init_lock=None,
            compute_all_metrics=False, activation_type='delta_epsilon', y1=0.05,
            track_gradients=True,
            use_fp8_inference=False
        )

        results['gcn'] = {
            'test_acc': gcn_result,
            'training_time': gcn_stats['training_time'],
            'gradient_variances': gcn_stats['gradient_variances'],
            'epoch_times': gcn_stats['epoch_times'],
            'num_epochs': gcn_stats['final_epoch'],
            'signal_gains': gcn_stats.get('signal_gains', [])
        }
    except Exception as e:
        print(f"GCN测试失败: {e}")
        results['gcn'] = {
            'test_acc': 0.0,
            'training_time': 0.0,
            'gradient_variances': [],
            'epoch_times': [],
            'num_epochs': 0,
            'signal_gains': []
        }

    # 测试GAT
    print("--- Testing GAT ---")
    try:
        gat_result, gat_stats = train_and_evaluate(
            'gat', 0, 0, 0,  # GAT不需要这些参数
            features_gpu, adj_gpu, labels_gpu, idx_train, idx_val, idx_test,
            lr, weight_decay, dropout, epochs, patience, nhid, nlayers, device,
            epsilon=0, c_activation=0, eta=0, w_threshold=0,
            verbose=False, fold_idx=0, total_folds=1, model_init_lock=None,
            compute_all_metrics=False, activation_type='delta_epsilon', y1=0.05,
            track_gradients=True,
            use_fp8_inference=False
        )

        results['gat'] = {
            'test_acc': gat_result,
            'training_time': gat_stats['training_time'],
            'gradient_variances': gat_stats['gradient_variances'],
            'epoch_times': gat_stats['epoch_times'],
            'num_epochs': gat_stats['final_epoch'],
            'signal_gains': gat_stats.get('signal_gains', [])
        }
    except Exception as e:
        print(f"GAT测试失败: {e}")
        results['gat'] = {
            'test_acc': 0.0,
            'training_time': 0.0,
            'gradient_variances': [],
            'epoch_times': [],
            'num_epochs': 0,
            'signal_gains': []
        }

    # 清理GPU数据
    del features_gpu, adj_gpu, labels_gpu, idx_train, idx_val, idx_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def plot_gradient_comparison(results, dataset_name, output_dir='results_v4'):
    """
    高级可视化：梯度方差对比和信号增益对比
    
    完全替换原版本，无需修改其他代码
    
    参数：
        results: dict，包含每个模型的统计数据
        dataset_name: str，数据集名称（用于文件名）
        output_dir: str，输出目录（默认 'results_v4'）
    
    返回：
        plot_path: str，保存的图表路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    opt_grad_vars = results['optimized']['gradient_variances']
    gcn_grad_vars = results['gcn']['gradient_variances']
    gat_grad_vars = results['gat']['gradient_variances']
    
    opt_signal_gains = results['optimized']['signal_gains']
    gcn_signal_gains = results['gcn']['signal_gains']
    gat_signal_gains = results['gat']['signal_gains']
    
    # 创建图表（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    # ==================== 左图：归一化梯度方差 ====================
    _plot_gradient_variance(ax1, opt_grad_vars, gcn_grad_vars, gat_grad_vars)
    
    # ==================== 右图：信号增益（带误差条） ====================
    _plot_signal_gain_errorbar(ax2, opt_signal_gains, gcn_signal_gains, gat_signal_gains)
    
    # 保存图表
    plt.tight_layout()
    plot_path = f"{output_dir}/{dataset_name}_gradient_variance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"[高级图表] 已保存至: {plot_path}")
    return plot_path


def _plot_gradient_variance(ax, opt_vars, gcn_vars, gat_vars):
    """
    绘制梯度方差曲线 - 左图
    """
    # 绘制LyapunovGCN
    if opt_vars:
        epochs = range(1, len(opt_vars) + 1)
        ax.plot(epochs, opt_vars,
                color=_COLOR_PALETTE['lyapunov'],
                label=_LABEL_CONFIG['lyapunov'],
                linewidth=2.8,
                alpha=0.85,
                zorder=3)

    # 绘制GCN
    if gcn_vars:
        epochs = range(1, len(gcn_vars) + 1)
        ax.plot(epochs, gcn_vars,
                color=_COLOR_PALETTE['gcn'],
                label=_LABEL_CONFIG['gcn'],
                linewidth=2.8, alpha=0.85, zorder=3)

    # 绘制GAT
    if gat_vars:
        epochs = range(1, len(gat_vars) + 1)
        ax.plot(epochs, gat_vars,
                color=_COLOR_PALETTE['gat'],
                label=_LABEL_CONFIG['gat'],
                linewidth=2.8, alpha=0.85, zorder=3)

    # 装饰左图
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Normalized Gradient Variance (Var / lr²)', 
                  fontsize=13, fontweight='bold', color='#2c3e50')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#2c3e50')
    
    legend = ax.legend(fontsize=11, loc='best', framealpha=0.95, 
                       edgecolor='#2c3e50', fancybox=True, shadow=True)
    legend.set_zorder(10)
    ax.set_facecolor('#f9f9f9')


def _plot_signal_gain_errorbar(ax, opt_gains, gcn_gains, gat_gains):
    """
    绘制信号增益曲线，带误差条（上下短线段） - 右图
    参考用户提供的图片样式
    """
    
    models_data = {
        'lyapunov': (opt_gains, _COLOR_PALETTE['lyapunov']),
        'gcn': (gcn_gains, _COLOR_PALETTE['gcn']),
        'gat': (gat_gains, _COLOR_PALETTE['gat'])
    }
    
    # 找到最大层数
    max_layers = max(
        len(opt_gains) if opt_gains else 0,
        len(gcn_gains) if gcn_gains else 0,
        len(gat_gains) if gat_gains else 0
    )
    
    if max_layers == 0:
        ax.text(0.5, 0.5, 'No signal gain data available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='#7f8c8d')
        return
    
    # x位置偏移（避免重叠）
    offset_map = {'lyapunov': -0.15, 'gcn': 0, 'gat': 0.15}
    
    # 绘制每个模型
    for model_key in ['lyapunov', 'gcn', 'gat']:
        gains, color = models_data[model_key]
        
        if not gains or len(gains) == 0:
            continue
        
        # 处理零值和负值
        gains_array = np.array([max(float(g), 1e-10) for g in gains])
        
        # 移除误差条 - signal_gains没有标准差信息
        # errors = gains_array * 50  # 注释掉，不使用人为的误差条
        
        # x坐标（带偏移）
        layers = np.arange(1, len(gains_array) + 1) + offset_map[model_key]
        
        # 不使用误差条 - signal_gains没有标准差信息
        
        # 绘制曲线
        ax.plot(layers, gains_array,
               color=color,
               label=_LABEL_CONFIG[model_key],
               linewidth=2.6,
               alpha=0.85,
               zorder=3)
        
        # 不使用散点标记 - 只显示光滑曲线
    
    # 装饰右图
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Signal Gain', fontsize=13, fontweight='bold', color='#2c3e50')

    ax.set_yscale('log')  # 对数坐标
    ax.set_ylim(1e-12, 1e20)  # 设置y轴范围从10^-9到10^12
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7,
            color='gray', which='both')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#2c3e50')

    # 设置横坐标刻度为10, 20, ..., 60
    xticks = list(range(10, 61, 10))  # 10, 20, 30, 40, 50, 60
    ax.set_xticks(xticks)
    
    legend = ax.legend(fontsize=11, loc='best', framealpha=0.95,
                       edgecolor='#2c3e50', fancybox=True, shadow=True)
    legend.set_zorder(10)
    ax.set_facecolor('#f9f9f9')


def main():
    # 设置PyTorch显存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

    parser = argparse.ArgumentParser(description='Version 4.0 Grid Search - 训练时归一化，推理时不归一化')
    parser.add_argument('--model', type=str, default='new_activation',
                        choices=['enhanced_gcnii', 'new_activation'],
                        help='模型类型 (default: new_activation)')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'squirrel', 'chameleon', 'texas'],
                        help='数据集: wisconsin, cora, citeseer, pubmed, squirrel, chameleon, physics, texas, cornell (default: cora)')
    parser.add_argument('--gamma_values', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.3,0.35,0.4,0.45,0.5],
                        help='gamma值列表 (default: 0.01, 0.05, 0.1, 0.2, 0.3)')
    parser.add_argument('--alpha_values', type=float, default=0.9,
                        help='alpha值列表 - NewActivation模型中控制激活函数比例 (default: 0.1~0.9)')
    parser.add_argument('--lambda_values', type=float, nargs='+',
                        default=[0.5,1,3,5,7,10],
                        help='lambda值列表 - 仅用于GCNII模型 (default: 0.5, 1.0, 1.5)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='激活函数阈值参数 (default: 0.1)')
    parser.add_argument('--c_activation', type=float, nargs='+', default=-0.484754509027,
                        help='激活函数sigmoid参数 (default: -0.484754509027)')
    parser.add_argument('--eta', type=float, default=0.0001,
                        help='权重矩阵单位混合系数 W_l = eta*I + (1-eta)*W_l (default: 0.0)')
    parser.add_argument('--w_threshold', type=float, default=0.0001,
                        help='权重阈值，小于此值的权重置为0，然后归一化 (default: 0.0，相当于ReLU)')
    parser.add_argument('--activation_type', type=str, default='three_segment',
                        choices=['delta_epsilon', 'piecewise_linear', 'three_segment'],
                        help='激活函数类型: delta_epsilon, piecewise_linear 或 three_segment (default: three_segment)')
    parser.add_argument('--y1', type=float, default=0.05,
                        help='分段线性激活函数的y1参数，第一个拐点(0.1, y1)的y坐标 (default: 0.05)')
    parser.add_argument('--x1', type=float, default=0.2,
                        help='三段激活函数的第一个阈值 x1 (default: 0.2)')
    parser.add_argument('--x2', type=float, default=0.9,
                        help='三段激活函数的第二个阈值 x2 (default: 0.9)')
    parser.add_argument('--layers', type=int, default=64,
                        help='层数 (default: 64)')
    parser.add_argument('--hidden', type=int, default=64,
                        help='隐藏层维度 (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率 (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减 (default: 5e-4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率 (default: 0.1)')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience (default: 100)')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='GPU ids (default: [0,1,2,3])')
    parser.add_argument('--k_folds', type=int, default=10,
                        help='K-fold (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                        help='输出详细的统计信息（第一个fold的所有层）')
    parser.add_argument('--run_baselines', nargs='*', choices=['GCN', 'GAT', 'APPNP', 'SGC', 'GCNII'],
                        help='运行基线模型，不指定参数则运行所有模型，可选: GCN, GAT, APPNP, SGC, GCNII (按顺序单GPU运行)')
    parser.add_argument('--use_all_metrics', action='store_true',
                        help='使用所有评价指标 (Accuracy, Macro-F1, Precision, Recall)')
    parser.add_argument('--compare_training', action='store_true',
                        help='运行优化模型、GCN和GAT的训练时间和梯度方差对比实验')
    parser.add_argument('--progressive_alpha', action='store_true',
                        help='启用alpha渐进模式：alpha从指定值逐层递增到1（仅对new_activation模型有效）')
    parser.add_argument('--benchmark_time', action='store_true',
                        help='运行时间基准测试：比较6个数据集在6个模型上的10折交叉验证总时间')
    parser.add_argument('--weight_sharing', action='store_true',
                        help='启用权重共享模式：每y层共享权重矩阵W')
    parser.add_argument('--sharing_interval', type=int, default=2,
                        help='权重共享间隔：每几层共享一个权重矩阵 (default: 2)')
    parser.add_argument('--doubly_stochastic', action='store_true',
                        help='使用双随机矩阵（行和列和都为1），默认使用列随机矩阵（仅列和为1）')
    parser.add_argument('--stochastic_iterations', type=int, default=20,
                        help='双随机矩阵归一化的最大迭代次数 (default: 20)')
    parser.add_argument('--use_fp8_inference', action='store_true',
                        help='在推理阶段使用FP8精度')
    parser.add_argument('--use_int8_inference', action='store_true',
                        help='在推理阶段使用INT8量化')
    
    args = parser.parse_args()
    
    # 设置随机种子确保可重复性
    # 这是全局种子，用于数据加载、fold划分等操作
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置确定性模式，确保CUDA操作可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("="*80)
    print("随机种子设置")
    print("="*80)
    print(f"全局随机种子: {seed}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"确定性模式: deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}")
    print("="*80 + "\n")
    
    # 创建results_v4目录和日志文件
    os.makedirs('results_v4', exist_ok=True)
    log_filename = f"results_v4/{args.model}_{args.dataset}_layers{args.layers}_eps{args.epsilon}_log.txt"
    
    # 重定向输出到文件和控制台
    tee = Tee(log_filename, mode='w')
    sys.stdout = tee
    
    # 检查GPU
    available_gpus = args.gpus if torch.cuda.is_available() else [None]
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
    
    # ==================== 时间基准测试模式 ====================
    if args.benchmark_time:
        print("\n" + "="*80)
        print("TIME BENCHMARK MODE ACTIVATED")
        print("="*80)
        print("Skipping grid search and running time benchmark across datasets...")
        print("="*80 + "\n")
        
        # 使用第一个GPU进行基准测试
        gpu_id = available_gpus[0]
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        
        # 运行时间基准测试
        benchmark_results = benchmark_time_across_datasets(args, device)
        
        # 保存结果到文件
        output_filename = f"results_v4/time_benchmark_layers{args.layers}.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("TIME BENCHMARK: 10-Fold Cross-Validation Total Time (seconds)\n")
            f.write("="*100 + "\n\n")
            
            # 表头
            header = f"{'Dataset':<15}"
            models = ['GCN', 'GAT', 'APPNP', 'SGC', 'GCNII', 'NewActivationGNN']
            for model in models:
                header += f"{model:>12}"
            header += f"{'Max Speedup':>15}"
            f.write(header + "\n")
            f.write("-" * 100 + "\n")
            
            # 数据行
            for dataset_name in ['texas', 'cora', 'chameleon', 'citeseer', 'squirrel', 'pubmed']:
                row = f"{dataset_name:<15}"
                times = []
                for model in models:
                    t = benchmark_results[dataset_name].get(model, -1)
                    if t > 0:
                        row += f"{t:>12.2f}"
                        times.append((model, t))
                    else:
                        row += f"{'FAIL':>12}"
                
                # 计算Speedup
                if len(times) > 0:
                    opt_time = benchmark_results[dataset_name].get('NewActivationGNN', -1)
                    if opt_time > 0:
                        slowest_time = max(t for _, t in times)
                        speedup = slowest_time / opt_time
                        row += f"{speedup:>15.2f}x"
                    else:
                        row += f"{'N/A':>15}"
                else:
                    row += f"{'N/A':>15}"
                
                f.write(row + "\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("Note:\n")
            f.write("  - Time: Total time for complete 10-fold cross-validation\n")
            f.write("  - Max Speedup: NewActivationGNN time / Slowest baseline time\n")
            f.write("  - Higher speedup means the optimized model is faster\n")
            f.write("="*100 + "\n")
        
        print(f"\nBenchmark results saved to: {output_filename}")
        print("\nBenchmark completed!")
        
        # 恢复标准输出并关闭日志文件
        original_stdout = tee.stdout
        sys.stdout = original_stdout
        tee.close()
        print(f"日志文件已关闭: {log_filename}")
        
        return  # 结束程序
    # =========================================================
    
    # ==================== 权重共享实验模式 ====================
    if args.weight_sharing:
        print("\n" + "="*80)
        print("WEIGHT SHARING MODE ACTIVATED")
        print("="*80)
        print("Skipping grid search and running weight sharing experiment...")
        print("="*80 + "\n")
        
        # 加载数据
        adj, features, labels, _, _, _ = load_data(args.dataset, use_standard_split=True, seed=seed)
        num_nodes = features.shape[0]
        all_indices = np.arange(num_nodes)
        
        # 使用第一个GPU进行实验
        gpu_id = available_gpus[0]
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        
        # 运行权重共享实验
        ws_results = run_weight_sharing_experiment(features, adj, labels, all_indices, args, device)
        
        # 保存结果到文件
        output_filename = f"results_v4/weight_sharing_{args.dataset}_layers{args.layers}_interval{args.sharing_interval}.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("权重共享实验结果\n")
            f.write("="*80 + "\n")
            f.write(f"数据集: {args.dataset}\n")
            f.write(f"层数: {args.layers}\n")
            f.write(f"权重共享间隔: 每 {args.sharing_interval} 层\n")
            f.write("="*80 + "\n\n")
            
            opt_acc, opt_std = ws_results['NewActivationGNN']['accuracy']
            
            for model_name in ['GCN', 'GAT', 'GCNII', 'NewActivationGNN']:
                if model_name in ws_results:
                    acc, std = ws_results[model_name]['accuracy']
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  准确率: {acc:.4f} ± {std:.4f}\n")
                    
                    if model_name != 'NewActivationGNN' and acc > 0:
                        improvement = ((opt_acc - acc) / acc * 100)
                        f.write(f"  优化模型提升: {improvement:+.2f}%\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n权重共享实验结果已保存到: {output_filename}")
        print("\n实验完成!")
        
        # 恢复标准输出并关闭日志文件
        original_stdout = tee.stdout
        sys.stdout = original_stdout
        tee.close()
        print(f"日志文件已关闭: {log_filename}")
        
        return  # 结束程序
    # =========================================================
    
    # 加载数据
    print("="*80)
    print("Version 4.0 - 训练时归一化，推理时不归一化")
    print("="*80)
    print("加载数据...")
    adj, features, labels, _, _, _ = load_data(args.dataset, use_standard_split=True, seed=seed)
    
    num_nodes = features.shape[0]
    all_indices = np.arange(num_nodes)
    
    print(f"数据集: {args.dataset}")
    print(f"节点数: {num_nodes}")
    print(f"特征维度: {features.shape[1]}")
    print(f"类别数: {int(labels.max().item() + 1)}")
    print(f"K-fold: {args.k_folds}")
    print(f"模型类型: {args.model}")
    if args.model == 'new_activation':
        print(f"激活函数类型: {args.activation_type}")
        if args.activation_type == 'delta_epsilon':
            print(f"Epsilon: {args.epsilon}")
            print(f"C_activation: {args.c_activation}")
        elif args.activation_type == 'piecewise_linear':
            print(f"Y1 (第一个拐点): {args.y1}")
            print(f"Y2 (第二个拐点): {1 - args.y1:.4f} (自动计算为 1-y1)")
    print(f"可用GPU: {available_gpus}")
    print("="*80 + "\n")
    
    # 生成实验配置
    experiments = []
    if args.model == 'new_activation':
        # NewActivation模型：网格搜索gamma、alpha和c_activation
        # 确保c_activation是列表
        if isinstance(args.c_activation, list):
            c_activation_list = args.c_activation
        else:
            c_activation_list = [args.c_activation]
        
        for gamma_val in args.gamma_values:
            alpha_val = 0.9
            for c_val in c_activation_list:
                experiments.append((gamma_val, alpha_val, 0.5, c_val))  # lambda_val=0.5（不使用）
        print(f"实验总数: {len(experiments)}")
        print(f"Gamma值: {args.gamma_values}")
        print(f"Alpha值: 0.9")
        print(f"C_activation值: {c_activation_list}")
    else:
        # EnhancedGCNII模型：网格搜索gamma、alpha和lambda
        for gamma_val in args.gamma_values:
            alpha_val = 0.9
            for lambda_val in args.lambda_values:
                experiments.append((gamma_val, alpha_val, lambda_val, None))  # 添加None作为c_val
        print(f"实验总数: {len(experiments)}")
        print(f"Gamma值: {args.gamma_values}")
        print(f"Alpha值: 0.9")
        print(f"Lambda值: {args.lambda_values}")
    print("="*80 + "\n")
    
    # 线程安全输出
    print_lock = threading.Lock()
    
    # 模型初始化锁，确保模型初始化是串行的，避免随机状态冲突
    model_init_lock = threading.Lock()
    
    # 单个实验执行函数
    def run_single_experiment(exp_idx, gamma_val, alpha_val, lambda_val, c_val, gpu_id):
        # 为每个实验设置确定性种子，确保相同参数产生相同结果
        # 使用固定种子42，不依赖exp_idx，这样相同的超参数配置总是产生相同的结果
        exp_seed = 42
        random.seed(exp_seed)
        np.random.seed(exp_seed)
        torch.manual_seed(exp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(exp_seed)
            torch.cuda.manual_seed_all(exp_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        
        adj_gpu = adj.to(device)
        features_gpu = features.to(device)
        labels_gpu = labels.to(device)
        
        model_name = "NewActivation" if args.model == 'new_activation' else "Enhanced GCNII"
        
        with print_lock:
            if args.model == 'new_activation':
                print(f"[GPU {gpu_id}] [{exp_idx+1}/{len(experiments)}] {model_name}: gamma={gamma_val:.3f}, alpha={alpha_val:.2f}, c={c_val:.2f}")
            else:
                print(f"[GPU {gpu_id}] [{exp_idx+1}/{len(experiments)}] {model_name}: gamma={gamma_val:.3f}, alpha={alpha_val:.2f}, lambda={lambda_val:.2f}")
        
        try:
            # 使用单个c_val值而不是整个列表
            result = k_fold_cv(args.model, gamma_val, alpha_val, lambda_val,
                              features_gpu, adj_gpu, labels_gpu, all_indices,
                              args.lr, args.weight_decay, args.dropout,
                              args.epochs, args.patience, args.hidden,
                              args.layers, device, k=args.k_folds,
                              epsilon=args.epsilon, c_activation=c_val if c_val is not None else -1.0,
                              eta=args.eta, w_threshold=args.w_threshold, verbose=args.verbose, 
                              model_init_lock=model_init_lock, compute_all_metrics=args.use_all_metrics,
                              activation_type=args.activation_type, y1=args.y1,
                              progressive_alpha=args.progressive_alpha, x1=args.x1, x2=args.x2,
                              doubly_stochastic=args.doubly_stochastic,
                              use_fp8_inference=args.use_fp8_inference,
                              use_int8_inference=args.use_int8_inference,
                              stochastic_iterations=args.stochastic_iterations)
            
            if args.use_all_metrics:
                with print_lock:
                    print(f"[GPU {gpu_id}] [{exp_idx+1}/{len(experiments)}] 完成:")
                    print(f"  Accuracy: {result['accuracy'][0]:.4f} ± {result['accuracy'][1]:.4f}")
                    print(f"  Macro-F1: {result['macro_f1'][0]:.4f} ± {result['macro_f1'][1]:.4f}")
                    print(f"  Macro-Precision: {result['macro_precision'][0]:.4f} ± {result['macro_precision'][1]:.4f}")
                    print(f"  Macro-Recall: {result['macro_recall'][0]:.4f} ± {result['macro_recall'][1]:.4f}")
                
                return {
                    'gamma': gamma_val,
                    'alpha': alpha_val,
                    'lambda': lambda_val,
                    'c_activation': c_val,
                    'metrics': result
                }
            else:
                mean_acc, std_acc = result
                with print_lock:
                    print(f"[GPU {gpu_id}] [{exp_idx+1}/{len(experiments)}] 完成: {mean_acc:.4f} ± {std_acc:.4f}")
                
                return {
                    'gamma': gamma_val,
                    'alpha': alpha_val,
                    'lambda': lambda_val,
                    'c_activation': c_val,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc
                }
        except Exception as e:
            with print_lock:
                print(f"[GPU {gpu_id}] [{exp_idx+1}/{len(experiments)}] 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            if args.use_all_metrics:
                return {
                    'gamma': gamma_val,
                    'alpha': alpha_val,
                    'lambda': lambda_val,
                    'c_activation': c_val,
                    'metrics': {
                        'accuracy': (0.0, 0.0),
                        'macro_f1': (0.0, 0.0),
                        'macro_precision': (0.0, 0.0),
                        'macro_recall': (0.0, 0.0)
                    },
                    'error': str(e)
                }
            else:
                return {
                    'gamma': gamma_val,
                    'alpha': alpha_val,
                    'lambda': lambda_val,
                    'c_activation': c_val,
                    'mean_acc': 0.0,
                    'std_acc': 0.0,
                    'error': str(e)
                }
    
    # 并行运行实验
    results = []
    print("开始并行实验...")
    print("="*80 + "\n")
    
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        future_to_exp = {}

        for i, (gamma_val, alpha_val, lambda_val, c_val) in enumerate(experiments):
            gpu_id = available_gpus[i % len(available_gpus)]
            future = executor.submit(run_single_experiment, i, gamma_val, alpha_val, lambda_val, c_val, gpu_id)
            future_to_exp[future] = (i, gamma_val, alpha_val, lambda_val, c_val)
        
        # 收集结果并定期清理内存
        for future in as_completed(future_to_exp):
            result = future.result()
            results.append(result)
            # 从字典中移除已完成的future，释放引用
            del future_to_exp[future]

            # 每处理完一定数量的实验就清理一次GPU内存
            if len(results) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 网格搜索完成后清理所有GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n网格搜索完成，已清理GPU缓存")
    
    # 获取最佳优化模型性能（用于与基线模型对比）
    if args.use_all_metrics:
        best_optimized = max(results, key=lambda x: x['metrics']['accuracy'][0])
        best_optimized_acc = best_optimized['metrics']['accuracy'][0]
        best_optimized_f1 = best_optimized['metrics']['macro_f1'][0]
        best_optimized_precision = best_optimized['metrics']['macro_precision'][0]
        best_optimized_recall = best_optimized['metrics']['macro_recall'][0]
    else:
        best_optimized = max(results, key=lambda x: x['mean_acc'])
        best_optimized_acc = best_optimized['mean_acc']

    # 从最佳配置中提取参数（用于输出）
    best_gamma = best_optimized['gamma']
    best_alpha = best_optimized['alpha']
    best_lambda = best_optimized.get('lambda', 0.5)
    best_c_activation = best_optimized.get('c_activation', -1.0)

    # 运行训练对比实验（如果启用）
    comparison_results = None
    if args.compare_training:
        print("\n" + "="*80)
        print("Running Training Efficiency Comparison...")
        print("="*80)

        # 使用第一个GPU进行对比实验
        gpu_id = available_gpus[0]
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')

        comparison_results = compare_models_training(
            features, adj, labels, all_indices,
            args.lr, args.weight_decay, args.dropout,
            args.epochs, args.patience, args.hidden, args.layers,
            device, args.epsilon, args.eta, args.w_threshold,
            best_gamma, best_alpha, best_lambda, best_c_activation,
            activation_type=args.activation_type, y1=args.y1,
            progressive_alpha=args.progressive_alpha, x1=args.x1, x2=args.x2,
            doubly_stochastic=args.doubly_stochastic,
            stochastic_iterations=args.stochastic_iterations
        )

        # 生成可视化
        plot_path = plot_gradient_comparison(comparison_results, args.dataset)
        print(f"\nGradient variance comparison plot saved to: {plot_path}")

        # 输出对比结果
        print("\n" + "="*80)
        print("Training Efficiency Comparison Results")
        print("="*80 + "\n")

        opt_data = comparison_results['optimized']
        gcn_data = comparison_results['gcn']
        gat_data = comparison_results['gat']

        # 时间对比
        print("=" * 70)
        print("TIME METRICS COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<40} {'Optimized':<15} {'GCN':<15} {'GAT':<15}")
        print("-" * 70)
        print(f"{'Total Training Time (s)':<40} {float(opt_data['training_time']):>14.2f} {float(gcn_data['training_time']):>14.2f} {float(gat_data['training_time']):>14.2f}")

        if opt_data['epoch_times'] and gcn_data['epoch_times'] and gat_data['epoch_times']:
            opt_avg_epoch_time = float(np.mean(opt_data['epoch_times'])) * 1000  # 转换为ms
            gcn_avg_epoch_time = float(np.mean(gcn_data['epoch_times'])) * 1000
            gat_avg_epoch_time = float(np.mean(gat_data['epoch_times'])) * 1000
            print(f"{'Avg Time per Epoch (ms)':<40} {opt_avg_epoch_time:>14.2f} {gcn_avg_epoch_time:>14.2f} {gat_avg_epoch_time:>14.2f}")

        print(f"{'Number of Epochs':<40} {int(opt_data['num_epochs']):>14d} {int(gcn_data['num_epochs']):>14d} {int(gat_data['num_epochs']):>14d}")

        gcn_time_ratio = float(gcn_data['training_time']) / float(opt_data['training_time']) if float(opt_data['training_time']) > 0 else 0.0
        gat_time_ratio = float(gat_data['training_time']) / float(opt_data['training_time']) if float(opt_data['training_time']) > 0 else 0.0
        print(f"{'Time Ratio vs Optimized':<40} {1.00:>14.2f} {gcn_time_ratio:>14.2f} {gat_time_ratio:>14.2f}")

        # 梯度方差对比
        print("\n" + "=" * 70)
        print("GRADIENT METRICS (Average Across All GCN Layers)")
        print("=" * 70)
        print(f"{'Metric':<40} {'Optimized':<15} {'GCN':<15} {'GAT':<15}")
        print("-" * 70)

        if opt_data['gradient_variances'] and gcn_data['gradient_variances'] and gat_data['gradient_variances']:
            opt_avg_grad_var = float(np.mean(opt_data['gradient_variances']))
            gcn_avg_grad_var = float(np.mean(gcn_data['gradient_variances']))
            gat_avg_grad_var = float(np.mean(gat_data['gradient_variances']))
            opt_std_grad_var = float(np.std(opt_data['gradient_variances']))
            gcn_std_grad_var = float(np.std(gcn_data['gradient_variances']))
            gat_std_grad_var = float(np.std(gat_data['gradient_variances']))

            print(f"{'Normalized Grad Var (mean)':<40} {opt_avg_grad_var:>14.6e} {gcn_avg_grad_var:>14.6e} {gat_avg_grad_var:>14.6e}")
            print(f"{'Normalized Grad Var (std)':<40} {opt_std_grad_var:>14.6e} {gcn_std_grad_var:>14.6e} {gat_std_grad_var:>14.6e}")

            gcn_grad_var_ratio = float(gcn_avg_grad_var) / float(opt_avg_grad_var) if float(opt_avg_grad_var) > 0 else 0.0
            gat_grad_var_ratio = float(gat_avg_grad_var) / float(opt_avg_grad_var) if float(opt_avg_grad_var) > 0 else 0.0
            print(f"{'Grad Var Ratio vs Optimized':<40} {1.00:>14.6f} {gcn_grad_var_ratio:>14.6f} {gat_grad_var_ratio:>14.6f}")

            print(f"\nNote: Normalized Gradient Variance = Var[∂L/∂W] / (learning_rate²)")
            print(f"      This metric is learning-rate independent and reflects gradient stability.")
        
        # 信号增益对比
        print("\n" + "=" * 70)
        print("SIGNAL GAIN METRICS (Layer-wise Weight Product Analysis)")
        print("=" * 70)
        print(f"{'Metric':<40} {'LyapunovGCN':<15} {'GCN':<15} {'GAT':<15}")
        print("-" * 70)
        
        # 检查是否有任何模型有信号增益数据
        has_signal_gains = (len(opt_data.get('signal_gains', [])) > 0 or 
                           len(gcn_data.get('signal_gains', [])) > 0 or 
                           len(gat_data.get('signal_gains', [])) > 0)
        
        if has_signal_gains:
            # 计算各模型的信号增益统计
            opt_gains = opt_data.get('signal_gains', [])
            gcn_gains = gcn_data.get('signal_gains', [])
            gat_gains = gat_data.get('signal_gains', [])
            
            opt_avg = float(np.mean(opt_gains)) if len(opt_gains) > 0 else 0.0
            gcn_avg = float(np.mean(gcn_gains)) if len(gcn_gains) > 0 else 0.0
            gat_avg = float(np.mean(gat_gains)) if len(gat_gains) > 0 else 0.0
            
            opt_std = float(np.std(opt_gains)) if len(opt_gains) > 0 else 0.0
            gcn_std = float(np.std(gcn_gains)) if len(gcn_gains) > 0 else 0.0
            gat_std = float(np.std(gat_gains)) if len(gat_gains) > 0 else 0.0
            
            print(f"{'Signal Gain (mean)':<40} {opt_avg:>14.6f} {gcn_avg:>14.6f} {gat_avg:>14.6f}")
            print(f"{'Signal Gain (std)':<40} {opt_std:>14.6f} {gcn_std:>14.6f} {gat_std:>14.6f}")
            print(f"{'Number of Layers':<40} {len(opt_gains):>14d} {len(gcn_gains):>14d} {len(gat_gains):>14d}")
            
            print(f"\nNote: Signal Gain = max(sum(|W_{{L-1}} * W_{{L-2}} * ... * W_l|, axis=1))")
            print(f"      This metric measures the maximum amplification from layer l to output.")
        else:
            print("No signal gain data available (check weight extraction)")
        
        print("\n" + "="*80 + "\n")

    # GCNII和其他基线模型地位平等，只有在--run_baselines中明确指定时才运行
    # 不再自动运行GCNII基准
    run_gcnii_baseline = False

    # 初始化变量，避免后续UnboundLocalError
    saved_baseline_acc = (0.5, 0.0)
    saved_baseline_f1 = (0.5, 0.0)
    saved_baseline_prec = (0.5, 0.0)
    saved_baseline_rec = (0.5, 0.0)

    # 初始化baseline_mean和baseline_std，避免在单指标模式下未定义
    baseline_mean = 0.5
    baseline_std = 0.0

    if run_gcnii_baseline:
        print("\n" + "="*80)
        print("运行标准GCNII基线（使用最佳alpha和lambda）...")
        print("="*80 + "\n")
        
        # 保存优化模型的最佳配置（用于输出）
        optimized_gamma = best_gamma
        optimized_alpha = best_alpha
        optimized_lambda = best_lambda
        optimized_c_activation = best_c_activation

        # 找出最佳配置用于GCNII基准
        if args.use_all_metrics:
            best_enhanced = max(results, key=lambda x: x['metrics']['accuracy'][0])
        else:
            best_enhanced = max(results, key=lambda x: x['mean_acc'])
        best_gamma = best_enhanced['gamma']
        best_alpha = best_enhanced['alpha']
        best_lambda = best_enhanced['lambda']
        best_c_activation = best_enhanced.get('c_activation', -1.0)
        
        # 获取优化模型的最佳性能用于对比
        if args.use_all_metrics:
            best_model_acc = best_enhanced['metrics']['accuracy'][0]
            best_model_f1 = best_enhanced['metrics']['macro_f1'][0]
            best_model_precision = best_enhanced['metrics']['macro_precision'][0]
            best_model_recall = best_enhanced['metrics']['macro_recall'][0]
        else:
            best_model_acc = best_enhanced['mean_acc']

        gpu_id = available_gpus[0]
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        adj_gpu = adj.to(device)
        features_gpu = features.to(device)
        labels_gpu = labels.to(device)

        # 初始化baseline_result变量，避免后续UnboundLocalError
        baseline_result = None
        
        if args.model == 'new_activation':
            print(f"[GPU {gpu_id}] 使用最佳配置: gamma={best_gamma:.3f}, alpha={best_alpha:.2f}, c={best_c_activation:.2f}")
        else:
            print(f"[GPU {gpu_id}] 使用最佳配置: gamma={best_gamma:.3f}, alpha={best_alpha:.2f}, lambda={best_lambda:.2f}")
        
        # 使用单个c值运行基线
        try:
            baseline_result = k_fold_cv('standard_gcnii', 0, best_alpha, best_lambda,
                                        features_gpu, adj_gpu, labels_gpu, all_indices,
                                        args.lr, args.weight_decay, args.dropout,
                                        args.epochs, args.patience, args.hidden,
                                        args.layers, device, k=args.k_folds,
                                        epsilon=args.epsilon, c_activation=-1.0, eta=0.0,
                                        w_threshold=0.0, verbose=False, model_init_lock=None,
                                        compute_all_metrics=args.use_all_metrics,
                                        use_fp8_inference=args.use_fp8_inference,
                                        use_int8_inference=args.use_int8_inference)
        except Exception as e:
            print(f"运行GCNII基线时出错: {e}")
            # 如果GCNII基线运行失败，提供默认值
            if args.use_all_metrics:
                baseline_result = {
                    'accuracy': (0.5, 0.0),
                    'macro_f1': (0.5, 0.0),
                    'macro_precision': (0.5, 0.0),
                    'macro_recall': (0.5, 0.0)
                }
            else:
                baseline_result = (0.5, 0.0)
        
        if args.use_all_metrics:
            baseline_acc_mean, baseline_acc_std = baseline_result['accuracy']
            baseline_f1_mean, baseline_f1_std = baseline_result['macro_f1']
            baseline_prec_mean, baseline_prec_std = baseline_result['macro_precision']
            baseline_rec_mean, baseline_rec_std = baseline_result['macro_recall']

            acc_improve = ((best_model_acc - baseline_acc_mean) / baseline_acc_mean * 100) if baseline_acc_mean > 0 else 0
            f1_improve = ((best_model_f1 - baseline_f1_mean) / baseline_f1_mean * 100) if baseline_f1_mean > 0 else 0
            prec_improve = ((best_model_precision - baseline_prec_mean) / baseline_prec_mean * 100) if baseline_prec_mean > 0 else 0
            rec_improve = ((best_model_recall - baseline_rec_mean) / baseline_rec_mean * 100) if baseline_rec_mean > 0 else 0

            print(f"标准GCNII:")
            print(f"  准确率: {baseline_acc_mean:.4f} ± {baseline_acc_std:.4f}  (优化模型提升: {acc_improve:+.2f}%)")
            print(f"  Macro-F1: {baseline_f1_mean:.4f} ± {baseline_f1_std:.4f}  (优化模型提升: {f1_improve:+.2f}%)")
            print(f"  Macro-Precision: {baseline_prec_mean:.4f} ± {baseline_prec_std:.4f}  (优化模型提升: {prec_improve:+.2f}%)")
            print(f"  Macro-Recall: {baseline_rec_mean:.4f} ± {baseline_rec_std:.4f}  (优化模型提升: {rec_improve:+.2f}%)")

            # 保存给后续使用
            baseline_mean, baseline_std = baseline_acc_mean, baseline_acc_std
        else:
            if isinstance(baseline_result, tuple) and len(baseline_result) == 2:
                baseline_mean, baseline_std = baseline_result
            else:
                # 如果baseline_result不是元组格式，使用默认值
                baseline_mean, baseline_std = 0.5, 0.0
            acc_improve = ((best_model_acc - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            print(f"标准GCNII: {baseline_mean:.4f} ± {baseline_std:.4f}  (优化模型提升: {acc_improve:+.2f}%)")

        # 保存baseline_result的值供后续输出使用
        if args.use_all_metrics:
            saved_baseline_acc = baseline_acc_mean, baseline_acc_std
            saved_baseline_f1 = baseline_f1_mean, baseline_f1_std
            saved_baseline_prec = baseline_prec_mean, baseline_prec_std
            saved_baseline_rec = baseline_rec_mean, baseline_rec_std
        else:
            saved_baseline_acc = baseline_mean, baseline_std

        # 清理GCNII基线模型的显存
        del adj_gpu, features_gpu, labels_gpu
        if 'baseline_result' in locals():
            del baseline_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"\n标准GCNII显存已清理")
    
    # 运行其他基线模型（如果启用）
    other_baselines = {}
    if args.run_baselines is not None:
        # 运行基线模型前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        print("运行基线模型前清理显存完成")
        # 如果没有指定具体模型，运行所有模型（包括GCNII）
        if len(args.run_baselines) == 0:
            baseline_models_to_run = None  # 运行所有模型（包括GCN, GAT, APPNP, SGC, GCNII）
            display_models = ['GCN', 'GAT', 'APPNP', 'SGC', 'GCNII']
        else:
            baseline_models_to_run = args.run_baselines
            display_models = baseline_models_to_run

        # 只有当有基线模型需要运行时，才调用train_baseline_models
        if display_models:
            gpu_id = available_gpus[0]  # 使用第一个GPU
            device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')

            print(f"\n{'='*80}")
            print(f"运行基线模型: {', '.join(display_models)}")
            print(f"{'='*80}\n")

            other_baselines = train_baseline_models(features, adj, labels, all_indices,
                                                     args.lr, args.weight_decay, args.dropout,
                                                     args.epochs, args.patience, args.hidden,
                                                     args.layers, device, k=args.k_folds,
                                                     compute_all_metrics=args.use_all_metrics,
                                                     baseline_models_to_run=baseline_models_to_run,
                                                     alpha=best_alpha,
                                                     use_int8_inference=args.use_int8_inference)
    
    # 排序结果
    if args.use_all_metrics:
        # 根据准确率排序
        if args.model == 'new_activation':
            results = sorted(results, key=lambda x: x['metrics']['accuracy'][0], reverse=True)
        else:
            results = sorted(results, key=lambda x: x['metrics']['accuracy'][0], reverse=True)
    else:
        if args.model == 'new_activation':
            results = sorted(results, key=lambda x: (x['gamma'], x['alpha'], x.get('c_activation', 0)))
        else:
            results = sorted(results, key=lambda x: (x['gamma'], x['alpha'], x['lambda']))
    
    # 准备输出
    output_lines = []
    output_lines.append("="*80)
    model_title = "NewActivation GNN (V4.0 - 训练时归一化)" if args.model == 'new_activation' else "Enhanced GCNII (V4.0)"
    output_lines.append(f"Version 4.0 - {model_title} 网格搜索结果")
    output_lines.append("="*80)
    output_lines.append(f"数据集: {args.dataset}")
    output_lines.append(f"层数: {args.layers}")
    output_lines.append(f"隐藏层维度: {args.hidden}")
    output_lines.append(f"K-fold: {args.k_folds}")
    output_lines.append(f"模型类型: {args.model}")
    if args.model == 'new_activation':
        output_lines.append(f"Epsilon: {args.epsilon}")
        output_lines.append(f"C_activation: {args.c_activation}")
    output_lines.append(f"可用GPU: {available_gpus}")
    output_lines.append("="*80)
    output_lines.append("")
    
    # 基线性能（只有当运行了GCNII基准时才显示）
    if run_gcnii_baseline:
        output_lines.append("="*80)
        output_lines.append("标准GCNII基线性能")
        output_lines.append("="*80)
        output_lines.append(f"配置: alpha={best_alpha:.2f}, lambda={best_lambda:.2f}")
        if args.use_all_metrics:
            # 使用保存的值
            acc_mean, acc_std = saved_baseline_acc
            f1_mean, f1_std = saved_baseline_f1
            prec_mean, prec_std = saved_baseline_prec
            rec_mean, rec_std = saved_baseline_rec

            acc_improve = ((best_model_acc - acc_mean) / acc_mean * 100) if acc_mean > 0 else 0
            f1_improve = ((best_model_f1 - f1_mean) / f1_mean * 100) if f1_mean > 0 else 0
            prec_improve = ((best_model_precision - prec_mean) / prec_mean * 100) if prec_mean > 0 else 0
            rec_improve = ((best_model_recall - rec_mean) / rec_mean * 100) if rec_mean > 0 else 0

            output_lines.append(f"准确率: {acc_mean:.4f} ± {acc_std:.4f}  (优化模型提升: {acc_improve:+.2f}%)")
            output_lines.append(f"Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}  (优化模型提升: {f1_improve:+.2f}%)")
            output_lines.append(f"Macro-Precision: {prec_mean:.4f} ± {prec_std:.4f}  (优化模型提升: {prec_improve:+.2f}%)")
            output_lines.append(f"Macro-Recall: {rec_mean:.4f} ± {rec_std:.4f}  (优化模型提升: {rec_improve:+.2f}%)")
        else:
            # 使用保存的值
            baseline_mean, baseline_std = saved_baseline_acc
            acc_improve = ((best_model_acc - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            output_lines.append(f"准确率: {baseline_mean:.4f} ± {baseline_std:.4f}  (优化模型提升: {acc_improve:+.2f}%)")
        output_lines.append("="*80)
        output_lines.append("")
    
    # 其他基线模型性能
    if args.run_baselines and other_baselines:
        output_lines.append("="*80)
        if run_gcnii_baseline:
            output_lines.append("其他基线模型性能对比")
        else:
            output_lines.append("基线模型性能")
        output_lines.append("="*80)
        for model_name, metrics in other_baselines.items():
            output_lines.append(f"\n{model_name}:")
            if args.use_all_metrics:
                acc_mean, acc_std = metrics['accuracy']
                f1_mean, f1_std = metrics['macro_f1']
                prec_mean, prec_std = metrics['macro_precision']
                rec_mean, rec_std = metrics['macro_recall']
                
                # 始终显示优化模型相对于基线模型的提升
                acc_improve = ((best_optimized_acc - acc_mean) / acc_mean * 100) if acc_mean > 0 else 0
                f1_improve = ((best_optimized_f1 - f1_mean) / f1_mean * 100) if f1_mean > 0 else 0
                prec_improve = ((best_optimized_precision - prec_mean) / prec_mean * 100) if prec_mean > 0 else 0
                rec_improve = ((best_optimized_recall - rec_mean) / rec_mean * 100) if rec_mean > 0 else 0
                
                output_lines.append(f"  准确率: {acc_mean:.4f} ± {acc_std:.4f}  (优化模型提升: {acc_improve:+.2f}%)")
                output_lines.append(f"  Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}  (优化模型提升: {f1_improve:+.2f}%)")
                output_lines.append(f"  Macro-Precision: {prec_mean:.4f} ± {prec_std:.4f}  (优化模型提升: {prec_improve:+.2f}%)")
                output_lines.append(f"  Macro-Recall: {rec_mean:.4f} ± {rec_std:.4f}  (优化模型提升: {rec_improve:+.2f}%)")
            else:
                acc_mean, acc_std = metrics['accuracy']
                # 始终显示优化模型相对于基线模型的提升
                acc_improve = ((best_optimized_acc - acc_mean) / acc_mean * 100) if acc_mean > 0 else 0
                output_lines.append(f"  准确率: {acc_mean:.4f} ± {acc_std:.4f}  (优化模型提升: {acc_improve:+.2f}%)")
        output_lines.append("="*80)
        output_lines.append("")
    
    # 性能对比
    output_lines.append("="*80)
    output_lines.append(f"{model_title} 性能对比")
    output_lines.append("="*80)
    
    if args.use_all_metrics:
        # 多指标表格
        if args.model == 'new_activation':
            header = f"{'Gamma':<8} {'Alpha':<8} {'C':<8} {'Accuracy':<18} {'Macro-F1':<18} {'Precision':<18} {'Recall':<18}"
        else:
            header = f"{'Gamma':<8} {'Alpha':<8} {'Lambda':<8} {'Accuracy':<18} {'Macro-F1':<18} {'Precision':<18} {'Recall':<18}"
        output_lines.append(header)
        output_lines.append("─"*110)
        
        for result in results:
            gamma_val = result['gamma']
            alpha_val = result['alpha']
            lambda_val = result['lambda']
            c_val = result.get('c_activation', None)
            metrics = result['metrics']
            
            if args.model == 'new_activation':
                line = f"{gamma_val:<8.3f} {alpha_val:<8.2f} {c_val:<8.2f} "
            else:
                line = f"{gamma_val:<8.3f} {alpha_val:<8.2f} {lambda_val:<8.2f} "
            
            line += f"{metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.4f}    "
            line += f"{metrics['macro_f1'][0]:.4f}±{metrics['macro_f1'][1]:.4f}    "
            line += f"{metrics['macro_precision'][0]:.4f}±{metrics['macro_precision'][1]:.4f}    "
            line += f"{metrics['macro_recall'][0]:.4f}±{metrics['macro_recall'][1]:.4f}"
            
            output_lines.append(line)
    else:
        # 单指标表格
        if args.model == 'new_activation':
            header = f"{'Gamma':<10} {'Alpha':<10} {'C':<10} {'测试准确率':<25}"
        else:
            header = f"{'Gamma':<10} {'Alpha':<10} {'Lambda':<10} {'测试准确率':<25}"
        output_lines.append(header)
        output_lines.append("─"*55)
        
        for result in results:
            gamma_val = result['gamma']
            alpha_val = result['alpha']
            lambda_val = result['lambda']
            c_val = result.get('c_activation', None)
            mean_acc = result['mean_acc']
            std_acc = result['std_acc']
            
            if args.model == 'new_activation':
                line = f"{gamma_val:<10.3f} {alpha_val:<10.2f} {c_val:<10.2f} {mean_acc:.4f} ± {std_acc:.4f}"
            else:
                line = f"{gamma_val:<10.3f} {alpha_val:<10.2f} {lambda_val:<10.2f} {mean_acc:.4f} ± {std_acc:.4f}"
            
            output_lines.append(line)
    
    output_lines.append("="*80)
    output_lines.append("")
    # 使用优化模型的最佳配置（如果运行了GCNII基准，则使用保存的配置）
    if run_gcnii_baseline:
        display_gamma = optimized_gamma
        display_alpha = optimized_alpha
        display_lambda = optimized_lambda
        display_c_activation = optimized_c_activation
    else:
        display_gamma = best_gamma
        display_alpha = best_alpha
        display_lambda = best_lambda
        display_c_activation = best_c_activation

    if args.model == 'new_activation':
        output_lines.append(f"最佳{model_title}配置: gamma={display_gamma:.3f}, alpha={display_alpha:.2f}, c={display_c_activation:.2f}")
    else:
        output_lines.append(f"最佳{model_title}配置: gamma={display_gamma:.3f}, alpha={display_alpha:.2f}, lambda={display_lambda:.2f}")
    
    if args.use_all_metrics:
        best_metrics = best_optimized['metrics']
        output_lines.append(f"最佳性能:")
        output_lines.append(f"  准确率: {best_metrics['accuracy'][0]:.4f} ± {best_metrics['accuracy'][1]:.4f}")
        output_lines.append(f"  Macro-F1: {best_metrics['macro_f1'][0]:.4f} ± {best_metrics['macro_f1'][1]:.4f}")
        output_lines.append(f"  Macro-Precision: {best_metrics['macro_precision'][0]:.4f} ± {best_metrics['macro_precision'][1]:.4f}")
        output_lines.append(f"  Macro-Recall: {best_metrics['macro_recall'][0]:.4f} ± {best_metrics['macro_recall'][1]:.4f}")
    else:
        output_lines.append(f"最佳性能: {best_optimized['mean_acc']:.4f} ± {best_optimized['std_acc']:.4f}")
    
    output_lines.append("")
    output_lines.append("实验完成！")
    
    # 输出到终端
    print("\n" + "="*80)
    for line in output_lines:
        print(line)
    
    # 保存到文件
    os.makedirs('results_v4', exist_ok=True)
    if args.run_baselines is not None and len(args.run_baselines) == 1:
        # 运行单个基线模型时，加上基线模型名称
        baseline_model = args.run_baselines[0]
        output_filename = f"results_v4/{args.dataset}_{baseline_model}_layers{args.layers}.txt"
    elif args.model == 'new_activation':
        output_filename = f"results_v4/{args.dataset}_newact_eps{args.epsilon}_layers{args.layers}.txt"
    else:
        output_filename = f"results_v4/{args.dataset}_layers{args.layers}.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n结果已保存到: {output_filename}")
    print(f"完整日志已保存到: {log_filename}")
    
    # 释放GPU内存和清理变量
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU内存已清理")

    # 清理不再需要的变量以释放内存
    try:
        del results, adj, features, labels, all_indices
        if 'baseline_results' in locals():
            del baseline_results
        import gc
        gc.collect()
        print("Python变量已清理")
    except NameError:
        pass

    # 恢复标准输出并关闭日志文件
    original_stdout = tee.stdout
    sys.stdout = original_stdout
    tee.close()
    print(f"日志文件已关闭: {log_filename}")


if __name__ == '__main__':
    main()
