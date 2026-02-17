import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datasets_v5 import load_data


def set_seed(seed):
    """设置随机种子以便复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_graph_data(dataset_name='cora'):
    """
    加载图数据集并计算对称归一化的邻接矩阵: D^{-1/2}AD^{-1/2}
    """
    dataset_name = dataset_name.lower()
    
    # 使用datasets_v5.py中的load_data函数从本地加载数据集
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    
    # adj已经是稀疏张量，转换为稠密矩阵
    if adj.is_sparse:
        A = adj.to_dense()
    else:
        A = adj
    
    # 获取节点数量
    num_nodes = A.shape[0]
    
    # 添加自环: A = A + I
    A = A + torch.eye(num_nodes)
    
    # 计算度矩阵D
    degree = A.sum(dim=1)
    
    # 计算 D^{-1/2}
    D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    
    # 计算对称归一化的邻接矩阵: D^{-1/2}AD^{-1/2}
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    # 获取特征维度
    if features.is_sparse:
        feature_dim = features.shape[1]
    else:
        feature_dim = features.shape[1]
    
    return A_norm, num_nodes, feature_dim


def he_initialization(in_features, out_features):
    """
    He (Kaiming) 初始化权重矩阵
    适用于ReLU激活函数
    """
    W = torch.empty(in_features, out_features)
    nn.init.kaiming_normal_(W, mode='fan_in', nonlinearity='relu')
    return W


def row_normalize(X, norm_type='l2', eps=1e-8):
    """
    行归一化（Row Normalization）
    对每个节点的特征向量进行归一化，除以该行的范数

    Args:
        X: 输入特征矩阵 (num_nodes, feature_dim)
        norm_type: 范数类型 ('l1', 'l2', 'inf')
        eps: 防止除零的小常数

    Returns:
        归一化后的特征矩阵
    """
    if norm_type == 'l1':
        # L1范数：各元素绝对值之和
        norms = torch.sum(torch.abs(X), dim=1, keepdim=True)
    elif norm_type == 'l2':
        # L2范数：欧几里得范数
        norms = torch.norm(X, p=2, dim=1, keepdim=True)
    elif norm_type == 'inf':
        # 无穷范数：最大绝对值
        norms = torch.max(torch.abs(X), dim=1, keepdim=True)[0]
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}. Choose from 'l1', 'l2', 'inf'")

    # 防止除零
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)

    return X / (norms + eps)


def gcn_layer_forward(A_norm, X, W, norm_type='l2'):
    """
    GCN层的前向传播
    X_{l+1} = ReLU(D^{-1/2}AD^{-1/2}X_lW)
    然后进行行归一化

    Args:
        A_norm: 对称归一化的邻接矩阵 D^{-1/2}AD^{-1/2}
        X: 当前层的特征矩阵
        W: 权重矩阵（固定不变）
        norm_type: 归一化使用的范数类型 ('l1', 'l2', 'inf')

    Returns:
        X_next: 下一层的特征矩阵
    """
    # X_{l+1} = ReLU(D^{-1/2}AD^{-1/2}X_lW)
    Z = A_norm @ X @ W
    X_next = F.relu(Z)

    # 行归一化
    X_next = row_normalize(X_next, norm_type=norm_type)

    return X_next


def iterate_gcn_to_convergence(A_norm, X_init, W, norm_type='l2', max_iters=10000, tolerance=1e-6,
                                check_interval=100, device='cuda'):
    """
    迭代GCN层直到收敛

    Args:
        A_norm: 对称归一化的邻接矩阵
        X_init: 初始特征矩阵
        W: 权重矩阵（固定不变）
        norm_type: 归一化使用的范数类型 ('l1', 'l2', 'inf')
        max_iters: 最大迭代次数
        tolerance: 收敛容差
        check_interval: 每隔多少次迭代记录一次变化
        device: 计算设备

    Returns:
        X_final: 最终收敛的特征矩阵
        convergence_history: 收敛历史记录
    """
    # 移动到设备
    A_norm = A_norm.to(device)
    W = W.to(device)
    X = X_init.to(device)
    
    convergence_history = {
        'iteration': [],
        'l1_change': [],
        'l2_change': [],
        'max_change': [],
        'mean': [],
        'std': []
    }
    
    print(f"  Starting GCN iteration...")
    print(f"  Initial X: mean={X.mean():.6f}, std={X.std():.6f}, min={X.min():.6f}, max={X.max():.6f}")
    
    for iter_num in range(max_iters):
        X_old = X.clone()
        
        # GCN层更新: X_{l+1} = RowNorm(ReLU(D^{-1/2}AD^{-1/2}X_lW))
        X = gcn_layer_forward(A_norm, X, W, norm_type=norm_type)
        
        # 计算变化量
        diff = X - X_old
        l1_change = torch.sum(torch.abs(diff)).item()
        l2_change = torch.norm(diff, p=2).item()
        max_change = torch.max(torch.abs(diff)).item()
        
        # 记录历史
        if iter_num % check_interval == 0:
            convergence_history['iteration'].append(iter_num)
            convergence_history['l1_change'].append(l1_change)
            convergence_history['l2_change'].append(l2_change)
            convergence_history['max_change'].append(max_change)
            convergence_history['mean'].append(X.mean().item())
            convergence_history['std'].append(X.std().item())
            
            print(f"  Iter {iter_num:5d}: L1={l1_change:.6f}, L2={l2_change:.6f}, "
                  f"Max={max_change:.6f}, Mean={X.mean():.6f}, Std={X.std():.6f}")
        
        # 检查收敛
        if l1_change < tolerance:
            print(f"  ✓ Converged at iteration {iter_num}!")
            convergence_history['iteration'].append(iter_num)
            convergence_history['l1_change'].append(l1_change)
            convergence_history['l2_change'].append(l2_change)
            convergence_history['max_change'].append(max_change)
            convergence_history['mean'].append(X.mean().item())
            convergence_history['std'].append(X.std().item())
            break
    else:
        print(f"  ✗ Did not converge after {max_iters} iterations")
        print(f"  Final L1 change: {l1_change:.6f} (tolerance: {tolerance})")
    
    print(f"  Final X: mean={X.mean():.6f}, std={X.std():.6f}, min={X.min():.6f}, max={X.max():.6f}")
    
    return X.cpu(), convergence_history


def plot_convergence_history(convergence_history, dataset_name, output_dir='results/gcn_convergence'):
    """
    绘制收敛历史曲线
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = convergence_history['iteration']
    
    # 绘制L1变化
    axes[0, 0].plot(iterations, convergence_history['l1_change'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('L1 Change', fontsize=12)
    axes[0, 0].set_title('L1 Norm of Change', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 绘制L2变化
    axes[0, 1].plot(iterations, convergence_history['l2_change'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('L2 Change', fontsize=12)
    axes[0, 1].set_title('L2 Norm of Change', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 绘制最大变化
    axes[1, 0].plot(iterations, convergence_history['max_change'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Max Change', fontsize=12)
    axes[1, 0].set_title('Maximum Element-wise Change', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 绘制均值和标准差
    ax2 = axes[1, 1]
    ax2.plot(iterations, convergence_history['mean'], 'b-', linewidth=2, label='Mean')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Mean', fontsize=12, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(iterations, convergence_history['std'], 'r-', linewidth=2, label='Std')
    ax2_twin.set_ylabel('Standard Deviation', fontsize=12, color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Mean and Std of X', fontsize=13, fontweight='bold')
    
    # 添加图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.suptitle(f'GCN Convergence Analysis - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'{output_dir}/convergence_plot_{dataset_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Convergence plot saved to: {filename}")
    plt.close()


def save_convergence_results(convergence_history, dataset_name, num_nodes, feature_dim, norm_type,
                            output_dir='results/gcn_convergence'):
    """
    保存收敛结果到文本文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{output_dir}/convergence_results_{dataset_name}.txt'
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"GCN CONVERGENCE TEST RESULTS - {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Update Rule: X_{{l+1}} = RowNorm(ReLU(D^{{-1/2}}AD^{{-1/2}}X_lW))\n")
        f.write(f"  Weight Initialization: He (Kaiming) initialization\n")
        f.write(f"  Weight Update: Fixed (no training)\n")
        f.write(f"  Normalization: Row normalization ({norm_type}-norm) after each layer\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Dataset: {dataset_name}\n")
        f.write(f"  Number of nodes: {num_nodes}\n")
        f.write(f"  Feature dimension: {feature_dim}\n\n")
        
        f.write("Convergence Summary:\n")
        iterations = convergence_history['iteration']
        if len(iterations) > 0:
            final_iter = iterations[-1]
            final_l1 = convergence_history['l1_change'][-1]
            final_l2 = convergence_history['l2_change'][-1]
            final_max = convergence_history['max_change'][-1]
            final_mean = convergence_history['mean'][-1]
            final_std = convergence_history['std'][-1]
            
            f.write(f"  Total iterations: {final_iter}\n")
            f.write(f"  Final L1 change: {final_l1:.8f}\n")
            f.write(f"  Final L2 change: {final_l2:.8f}\n")
            f.write(f"  Final max change: {final_max:.8f}\n")
            f.write(f"  Final mean: {final_mean:.8f}\n")
            f.write(f"  Final std: {final_std:.8f}\n\n")
        
        f.write("Detailed Convergence History:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Iteration':<12} {'L1 Change':<15} {'L2 Change':<15} {'Max Change':<15} {'Mean':<12} {'Std':<12}\n")
        f.write("-"*80 + "\n")
        
        for i in range(len(iterations)):
            f.write(f"{iterations[i]:<12} "
                   f"{convergence_history['l1_change'][i]:<15.8f} "
                   f"{convergence_history['l2_change'][i]:<15.8f} "
                   f"{convergence_history['max_change'][i]:<15.8f} "
                   f"{convergence_history['mean'][i]:<12.8f} "
                   f"{convergence_history['std'][i]:<12.8f}\n")
        
        f.write("="*80 + "\n")
    
    print(f"  Results saved to: {filename}")


def run_convergence_test(dataset_name, feature_dim=None, norm_type='l2', max_iters=10000, tolerance=1e-6,
                         check_interval=100, device='cuda', output_dir='results/gcn_convergence',
                         seed=42):
    """
    运行GCN收敛性测试
    """
    print(f"\n{'='*80}")
    print(f"GCN CONVERGENCE TEST - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 加载图数据
    print(f"Loading graph data...")
    A_norm, num_nodes, data_feature_dim = load_graph_data(dataset_name)
    print(f"  Nodes: {num_nodes}, Data feature dim: {data_feature_dim}")
    
    # 如果没有指定feature_dim，使用数据集的特征维度
    if feature_dim is None:
        feature_dim = data_feature_dim
    
    print(f"  Using feature dimension: {feature_dim}")
    
    # 初始化权重矩阵W（He初始化，然后固定不变）
    print(f"Initializing weight matrix with He initialization...")
    W = he_initialization(feature_dim, feature_dim)
    print(f"  W shape: {W.shape}, W mean: {W.mean():.6f}, W std: {W.std():.6f}")
    
    # 初始化特征矩阵X（随机初始化）
    print(f"Initializing feature matrix...")
    X_init = torch.randn(num_nodes, feature_dim)
    # 标准化初始特征
    X_init = row_normalize(X_init, norm_type=norm_type)
    print(f"  X_init shape: {X_init.shape}, X_init mean: {X_init.mean():.6f}, X_init std: {X_init.std():.6f}")
    
    # 迭代到收敛
    print(f"\nStarting convergence test...")
    print(f"  Max iterations: {max_iters}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Check interval: {check_interval}")
    print(f"  Row normalization: {norm_type}-norm")

    X_final, convergence_history = iterate_gcn_to_convergence(
        A_norm, X_init, W,
        norm_type=norm_type,
        max_iters=max_iters,
        tolerance=tolerance,
        check_interval=check_interval,
        device=device
    )
    
    # 绘制收敛曲线
    plot_convergence_history(convergence_history, dataset_name, output_dir)
    
    # 保存结果
    save_convergence_results(convergence_history, dataset_name, num_nodes, feature_dim, norm_type, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Test completed for {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    return X_final, convergence_history


def main():
    parser = argparse.ArgumentParser(description='GCN Convergence Test')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['cora', 'citeseer', 'pubmed'],
                       help='List of datasets to test')
    parser.add_argument('--feature_dim', type=int, default=None,
                       help='Feature dimension (default: use dataset feature dim)')
    parser.add_argument('--norm_type', type=str, default='l2', choices=['l1', 'l2', 'inf'],
                       help='Row normalization norm type (default: l2)')
    parser.add_argument('--max_iters', type=int, default=20000,
                       help='Maximum number of iterations (default: 10000)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--check_interval', type=int, default=100,
                       help='Interval to check and record convergence (default: 100)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results/gcn_convergence',
                       help='Output directory for results (default: results/gcn_convergence)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print("\n" + "="*80)
    print("GCN CONVERGENCE EXPERIMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Feature dimension: {args.feature_dim if args.feature_dim else 'auto (use dataset dim)'}")
    print(f"  Row normalization: {args.norm_type}-norm")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Tolerance: {args.tolerance}")
    print(f"  Check interval: {args.check_interval}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80)
    
    # 对每个数据集运行测试
    all_results = {}
    for dataset_name in args.datasets:
        try:
            X_final, convergence_history = run_convergence_test(
                dataset_name=dataset_name,
                feature_dim=args.feature_dim,
                norm_type=args.norm_type,
                max_iters=args.max_iters,
                tolerance=args.tolerance,
                check_interval=args.check_interval,
                device=device,
                output_dir=args.output_dir,
                seed=args.seed
            )
            all_results[dataset_name] = {
                'X_final': X_final,
                'convergence_history': convergence_history
            }
        except Exception as e:
            print(f"\nError processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    os.makedirs(args.output_dir, exist_ok=True)
    summary_file = f'{args.output_dir}/convergence_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GCN CONVERGENCE TEST - SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Feature dimension: {args.feature_dim if args.feature_dim else 'auto'}\n")
        f.write(f"  Max iterations: {args.max_iters}\n")
        f.write(f"  Tolerance: {args.tolerance}\n")
        f.write(f"  Random seed: {args.seed}\n\n")
        
        f.write("Results by Dataset:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Dataset':<15} {'Converged':<12} {'Iterations':<12} {'Final L1':<15} {'Final Mean':<12} {'Final Std':<12}\n")
        f.write("-"*80 + "\n")
        
        for dataset_name, result in all_results.items():
            history = result['convergence_history']
            if len(history['iteration']) > 0:
                final_iter = history['iteration'][-1]
                final_l1 = history['l1_change'][-1]
                final_mean = history['mean'][-1]
                final_std = history['std'][-1]
                converged = "Yes" if final_l1 < args.tolerance else "No"
                
                f.write(f"{dataset_name:<15} {converged:<12} {final_iter:<12} "
                       f"{final_l1:<15.8f} {final_mean:<12.6f} {final_std:<12.6f}\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"All tests completed!")
    print(f"Summary saved to: {summary_file}")
    print(f"Individual results saved to: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
