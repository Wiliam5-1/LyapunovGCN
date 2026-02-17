"""
完全INT8推理工具
支持GNN模型的完全INT8推理（权重+激活+激活函数全部INT8）
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import warnings

# 导入完全INT8推理引擎
from full_int8_inference import (
    convert_to_full_int8,
    FullINT8Model,
    INT8ActivationFunction,
    INT8TensorQuantizer,
    INT8MatMul
)

# INC已弃用，使用完全INT8推理引擎
INC_AVAILABLE = True  # 为了兼容性保持True


def quantize_model_to_int8(model: nn.Module, 
                           example_inputs: Tuple[torch.Tensor, ...],
                           backend='default',
                           activation_params=None) -> nn.Module:
    """
    完全INT8量化：权重、激活、激活函数全部INT8
    
    Args:
        model: PyTorch模型
        example_inputs: 示例输入 (features, adj)
        backend: 量化后端 (兼容参数，实际使用自定义INT8引擎)
        activation_params: 激活函数参数（可选）
        
    Returns:
        完全INT8的模型包装器
    """
    print("="*80)
    print("开始完全INT8量化（权重 + 激活 + 激活函数）")
    print("="*80)
    
    try:
        # 设置模型为评估模式
        model.eval()
        model.cpu()  # 量化在CPU上进行
        
        print("配置完全INT8量化策略...")
        print(f"  - 权重: INT8（所有层）")
        print(f"  - 激活: INT8（动态量化）")
        print(f"  - 激活函数: INT8域查找表计算")
        print(f"  - 矩阵乘法: INT8 @ INT8 -> INT32累积 -> INT8输出")
        
        # 如果模型有激活函数参数，提取它们（优化模型）
        if activation_params is None:
            activation_params = {}
            # 尝试从模型中提取激活函数参数（优化模型特有）
            if hasattr(model, 'x1'):
                activation_params['x1'] = model.x1
            if hasattr(model, 'x2'):
                activation_params['x2'] = model.x2
            if hasattr(model, 'c_activation'):
                activation_params['c_activation'] = model.c_activation
            if hasattr(model, 'epsilon'):
                activation_params['epsilon'] = model.epsilon
            if hasattr(model, 'activation_type'):
                activation_params['activation_type'] = model.activation_type
            
            # 如果没有提取到任何参数，设为None让convert_to_full_int8自动检测
            if not activation_params:
                activation_params = None
        
        # 使用完全INT8引擎进行量化
        # activation_params为None时，会自动检测基线模型的激活函数类型
        quantized_model = convert_to_full_int8(
            model, 
            example_inputs,
            activation_params=activation_params
        )
        
        print("\n" + "="*80)
        print("✓ 完全INT8量化完成")
        print("  已量化: 权重层 + 激活值 + 激活函数")
        print("  推理模式: 完全INT8（权重+激活+激活函数）")
        print("="*80 + "\n")
        
        return quantized_model
        
    except Exception as e:
        print(f"INT8量化失败: {e}")
        import traceback
        traceback.print_exc()
        print("返回原始模型（未量化）")
        return model


def evaluate_int8_model(model, features, adj, labels, idx, device='cpu'):
    """
    使用完全INT8模型进行推理评估
    
    Args:
        model: 量化后的模型（在CPU上）
        features: 特征矩阵
        adj: 邻接矩阵
        labels: 标签
        idx: 评估索引
        device: 原始设备（用于数据准备）
        
    Returns:
        损失和准确率
    """
    import torch.nn.functional as F
    
    # 将数据移到CPU（量化模型在CPU上运行）
    features_cpu = features.cpu()
    labels_cpu = labels.cpu()
    idx_cpu = idx.cpu()
    
    # 处理邻接矩阵
    if adj.is_sparse:
        adj_cpu = adj.cpu().to_dense()
    else:
        adj_cpu = adj.cpu()
    
    # INT8推理
    model.eval()
    with torch.no_grad():
        output = model(features_cpu, adj_cpu)
    
    # 计算损失和准确率
    loss = F.nll_loss(output[idx_cpu], labels_cpu[idx_cpu])
    pred = output[idx_cpu].max(1)[1]
    acc = pred.eq(labels_cpu[idx_cpu]).sum().item() / idx_cpu.size(0)
    
    return loss.item(), acc


# 向后兼容的别名
quantize_model_pytorch_fallback = quantize_model_to_int8


if __name__ == "__main__":
    print("完全INT8推理模块加载成功")
    print("支持：权重INT8 + 激活INT8 + 激活函数INT8")
