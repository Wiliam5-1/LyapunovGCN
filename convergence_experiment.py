import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
from itertools import combinations
from datasets_v5 import load_data

# 设置随机种子以便复现（但我们会在实验中改变它）
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def three_segment_activation(x, c=-0.484754509027, x1=0.2, x2=0.9):
    """
    三段激活函数（与grid_search_v6.py中的模型一致）
    δ(x) = 0, if x ∈ [0, x1]
         = θ * (1 + e^(-c)) / (1 + e^(-cθ)), if x ∈ (x1, x2)
         = 1, if x ∈ [x2, 1]
    其中 θ = (x - x1) / (x2 - x1)
    """
    # 创建三个区域的掩码
    mask1 = x <= x1
    mask2 = (x > x1) & (x < x2)
    mask3 = x >= x2
    
    # 第一段: 输出0
    segment1 = torch.zeros_like(x)
    
    # 第二段: θ * (1 + e^(-c)) / (1 + e^(-cθ))
    theta = (x - x1) / (x2 - x1 + 1e-8)
    theta = torch.clamp(theta, 0, 1)
    numerator = 1 + torch.exp(torch.tensor(-c, device=x.device))
    denominator = 1 + torch.exp(-c * theta)
    segment2 = theta * numerator / denominator
    
    # 第三段: 输出1
    segment3 = torch.ones_like(x)
    
    # 组合三个段落
    output = mask1 * segment1 + mask2 * segment2 + mask3 * segment3
    
    return output

def create_column_stochastic_matrix(n, m):
    """
    创建列随机矩阵（每列和为1）
    在[0,1]范围内随机初始化后进行列归一化
    """
    # 在[0,1]范围内随机初始化
    W = torch.rand(n, m)
    # 列归一化（每列和为1）
    W = W / W.sum(dim=0, keepdim=True)

    return W

def load_graph_data(dataset_name='cora'):
    """
    加载图数据集获得A和D
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
    
    # 添加自环
    A = A + torch.eye(num_nodes)
    
    # 计算度矩阵D
    degree = A.sum(dim=1)
    D_inv = torch.diag(1.0 / degree)
    
    # D^{-1}A 是行随机的
    D_inv_A = D_inv @ A
    
    return D_inv_A, num_nodes

def initialize_X(num_nodes, feature_dim):
    """
    初始化X在[0,1]之间的随机数，然后以0.6的dropout率随机失活（模拟稀疏特征），
    最后进行Min-max归一化确保范围在[0,1]
    """
    X = torch.rand(num_nodes, feature_dim)
    # 以0.6的dropout率随机失活（60%的元素变成0）
    X = torch.nn.functional.dropout(X, p=0, training=True)

    # Min-max归一化到[0,1]范围
    X_min = X.min()
    X_max = X.max()
    if X_max - X_min > 1e-8:  # 避免除零
        X = (X - X_min) / (X_max - X_min)
    else:
        X = torch.clamp(X, 0, 1)  # 如果所有值相等，设为[0,1]范围内的值

    return X

def iterate_to_equilibrium(D_inv_A, W, X_init, c=-0.484754509027, x1=0.2, x2=0.9,
                          max_iters=500000, tolerance=1e-6, device='cuda'):
    """
    迭代更新X直到达到平衡解
    更新公式: X_{l+1} = δ(D^{-1}AX_lW)
    注意: W在整个迭代过程中保持不变，只有X会更新
    返回: 平衡解、实际迭代次数、每次迭代的X序列
    """
    # 移动到设备
    D_inv_A = D_inv_A.to(device)
    W = W.to(device)
    X = X_init.to(device)

    # 保存每次迭代的X值（包括初始值）
    X_history = [X_init.cpu().clone()]

    for iter_num in range(max_iters):
        X_old = X.clone()

        # X_{l+1} = δ(D^{-1}AX_lW)
        Z = D_inv_A @ X @ W
        X_new = three_segment_activation(Z, c=c, x1=x1, x2=x2)

        # 计算变化（使用L1范数）
        change = torch.sum(torch.abs(X_new - X_old)).item()

        X = X_new

        # 保存当前迭代的X值
        X_history.append(X.cpu().clone())

        # 检查收敛
        if change < tolerance:
            break

    # 返回平衡解、实际迭代次数、X的历史序列
    actual_iters = iter_num + 1
    return X.cpu(), actual_iters, X_history

def compute_cosine_similarity_matrix(equilibria_list):
    """
    计算多个平衡解之间的余弦相似度矩阵
    
    Args:
        equilibria_list: List of equilibrium solutions (each is a tensor)
    
    Returns:
        similarity_matrix: n x n matrix of cosine similarities
    """
    n = len(equilibria_list)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 将矩阵展平为向量
            vec_i = equilibria_list[i].flatten()
            vec_j = equilibria_list[j].flatten()
            
            # 计算向量的范数
            norm_i = torch.norm(vec_i)
            norm_j = torch.norm(vec_j)
            
            # 处理全零向量的特殊情况
            if norm_i < 1e-10 and norm_j < 1e-10:
                # 两个都是全零向量，认为它们完全相同
                cos_sim = 1.0
            elif norm_i < 1e-10 or norm_j < 1e-10:
                # 只有一个是全零向量，认为它们完全不同
                cos_sim = 0.0
            else:
                # 正常情况，计算余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec_i.unsqueeze(0), vec_j.unsqueeze(0)
                ).item()
            
            similarity_matrix[i, j] = cos_sim
    
    return similarity_matrix

def format_matrix_for_output(matrix, max_display_rows=None, max_display_cols=None):
    """
    将矩阵格式化为易读的字符串
    如果 max_display_rows 和 max_display_cols 为 None，则显示完整矩阵
    """
    if matrix.numel() == 0:
        return "[]"

    rows, cols = matrix.shape
    output_lines = []

    # 显示矩阵形状
    output_lines.append(f"Shape: ({rows}, {cols})")

    # 如果没有指定显示限制，则显示完整矩阵
    if max_display_rows is None:
        display_rows = rows
    else:
        display_rows = min(rows, max_display_rows)

    if max_display_cols is None:
        display_cols = cols
    else:
        display_cols = min(cols, max_display_cols)

    for i in range(display_rows):
        row_data = matrix[i, :display_cols]
        if hasattr(row_data, 'numpy'):
            row_values = row_data.numpy()
        else:
            row_values = row_data

        # 格式化每一行
        formatted_values = []
        for val in row_values:
            formatted_values.append(f"{val:.4f}")

        row_str = "  ".join(formatted_values)
        if max_display_cols is not None and cols > max_display_cols:
            row_str += "  ..."
        output_lines.append(f"  {row_str}")

    if max_display_rows is not None and rows > max_display_rows:
        output_lines.append(f"  ... ({rows - max_display_rows} more rows)")

    return "\n".join(output_lines)

def save_trial_history_to_file(X_history, trial_num, dataset_name, output_dir='results'):
    """
    将单次实验的X迭代历史保存到txt文件
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/{dataset_name}_trial_{trial_num + 1}_history.txt'

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name.upper()}\n")
        f.write(f"Trial: {trial_num + 1}\n")
        f.write(f"Total iterations: {len(X_history)}\n")
        f.write("="*60 + "\n\n")

        for iter_num, X in enumerate(X_history):
            f.write(f"第{iter_num + 1}次迭代：X_{iter_num + 1}=\n")
            f.write(format_matrix_for_output(X, max_display_rows=None, max_display_cols=None))
            f.write("\n\n")

    return filename

def run_multiple_trials(D_inv_A, num_nodes, feature_dim, gamma=0.05,
                       num_trials=10, device='cuda', dataset_name='cora', output_dir='results'):
    """
    运行多次实验，每次使用不同的随机初始化
    每次trial创建新的W和X_init，然后在迭代过程中W保持不变，只有X更新
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Running {num_trials} trials with different random initializations...")
    
    equilibria_list = []
    iterations_list = []

    # 默认参数（与grid_search_v6.py保持一致）
    c = -0.484754509027
    x1 = 0.2
    x2 = 0.9

    for trial in range(num_trials):
        # 每次trial设置不同的随机种子
        set_seed(trial * 42)

        # 每次trial创建新的W矩阵（不同的随机初始化）
        W = create_column_stochastic_matrix(feature_dim, feature_dim)

        # 每次trial创建新的X初始化
        X_init = initialize_X(num_nodes, feature_dim)

        # 迭代到平衡解（在迭代过程中W保持不变，只有X更新）
        X_eq, actual_iters, X_history = iterate_to_equilibrium(
            D_inv_A, W, X_init,
            c=c, x1=x1, x2=x2,
            max_iters=20000, tolerance=1e-6, device=device
        )

        equilibria_list.append(X_eq)
        iterations_list.append(actual_iters)

        # 保存X的迭代历史到文件
        history_file = save_trial_history_to_file(X_history, trial, dataset_name, output_dir)
        print(f"  Trial {trial+1}/{num_trials}: Equilibrium reached in {actual_iters} iterations. "
              f"X range: [{X_eq.min():.4f}, {X_eq.max():.4f}], "
              f"X Mean: {X_eq.mean():.4f}, X Std: {X_eq.std():.4f}, "
              f"W Mean: {W.mean():.4f}")
        print(f"    History saved to: {history_file}")
    
    # 计算余弦相似度矩阵
    similarity_matrix = compute_cosine_similarity_matrix(equilibria_list)
    
    # 提取上三角（不包括对角线）的相似度值
    upper_triangle_indices = np.triu_indices(num_trials, k=1)
    pairwise_similarities = similarity_matrix[upper_triangle_indices]
    
    # 统计信息
    mean_similarity = np.mean(pairwise_similarities)
    std_similarity = np.std(pairwise_similarities)
    min_similarity = np.min(pairwise_similarities)
    max_similarity = np.max(pairwise_similarities)
    
    print(f"\n  Cosine Similarity Statistics:")
    print(f"    Mean: {mean_similarity:.6f}")
    print(f"    Std:  {std_similarity:.6f}")
    print(f"    Min:  {min_similarity:.6f}")
    print(f"    Max:  {max_similarity:.6f}")
    
    # 判断是否过平滑（如果相似度普遍很高，说明过平滑）
    threshold = 0.95
    over_smooth_ratio = np.sum(pairwise_similarities > threshold) / len(pairwise_similarities)
    
    print(f"\n  Over-smoothing Analysis:")
    print(f"    Similarity > {threshold}: {over_smooth_ratio*100:.2f}% of pairs")
    if over_smooth_ratio < 0.1:
        print(f"    ✓ NOT over-smoothed (diverse equilibria)")
    else:
        print(f"    ✗ Potentially over-smoothed (similar equilibria)")
    
    return similarity_matrix, pairwise_similarities, {
        'mean': mean_similarity,
        'std': std_similarity,
        'min': min_similarity,
        'max': max_similarity,
        'over_smooth_ratio': over_smooth_ratio
    }

def plot_similarity_heatmap(similarity_matrix, dataset_name, output_dir='results'):
    """
    绘制余弦相似度热图
    """
    plt.figure(figsize=(8, 6.5))
    
    # 使用seaborn绘制热图
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn_r',  # 红色表示高相似度，绿色表示低相似度
                vmin=0.0, 
                vmax=1.0,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5,
                linecolor='gray',
                mask=mask)
    
    plt.title(f'Equilibrium Cosine Similarity - {dataset_name.upper()}', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Trial Index', fontsize=12)
    plt.ylabel('Trial Index', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/similarity_heatmap_{dataset_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n  Heatmap saved to: {filename}")
    plt.close()

def plot_similarity_distribution(all_similarities_dict, output_dir='results'):
    """
    绘制所有数据集的相似度分布对比图（小提琴图）
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    datasets = ['cora', 'citeseer', 'pubmed', 'texas', 'chameleon', 'squirrel']
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        
        if dataset_name in all_similarities_dict:
            similarities = all_similarities_dict[dataset_name]
            stats = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            }
            
            # 绘制小提琴图
            parts = ax.violinplot([similarities], positions=[0], widths=0.7,
                                 showmeans=True, showmedians=True)
            
            # 设置颜色
            for pc in parts['bodies']:
                pc.set_facecolor('#8dd3c7')
                pc.set_alpha(0.7)
            
            # 添加统计信息文本
            textstr = f"Mean: {stats['mean']:.4f}\nStd: {stats['std']:.4f}\n" \
                     f"Min: {stats['min']:.4f}\nMax: {stats['max']:.4f}"
            ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            ax.set_title(f'{dataset_name.upper()}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cosine Similarity', fontsize=10)
            ax.set_ylim([0, 1])
            ax.set_xticks([])
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{dataset_name.upper()}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Equilibrium Diversity across Datasets\n(Lower similarity = More diverse = Less over-smoothing)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/similarity_distribution_all_datasets.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nCombined distribution plot saved to: {filename}")
    plt.close()

def plot_pairwise_comparison(all_similarities_dict, output_dir='results'):
    """
    绘制数据集之间的成对比较箱线图
    """
    datasets = ['cora', 'citeseer', 'pubmed', 'texas', 'chameleon', 'squirrel']
    data_to_plot = []
    labels = []
    
    for dataset_name in datasets:
        if dataset_name in all_similarities_dict:
            data_to_plot.append(all_similarities_dict[dataset_name])
            labels.append(dataset_name.upper())
    
    if len(data_to_plot) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 创建箱线图
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                   notch=True, showmeans=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   meanprops=dict(marker='D', markerfacecolor='green', markersize=6))
    
    # 添加参考线
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
              label='Over-smoothing Threshold (0.95)')
    
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Equilibrium Diversity Comparison across Datasets', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    filename = f'{output_dir}/similarity_boxplot_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Boxplot comparison saved to: {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Equilibrium Diversity Experiment')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['cora', 'citeseer', 'pubmed', 'texas', 'chameleon', 'squirrel'],
                       help='List of datasets to evaluate')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of trials with different random initializations (default: 10)')
    parser.add_argument('--feature_dim', type=int, default=32,
                       help='Feature dimension (default: 64)')
    parser.add_argument('--gamma', type=float, default=0.05,
                       help='Gamma parameter (default: 0.05)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print("\n" + "="*60)
    print("EQUILIBRIUM DIVERSITY EXPERIMENT")
    print("="*60)
    print(f"Parameters:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Number of trials: {args.num_trials}")
    print(f"  Feature dimension: {args.feature_dim}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Output directory: {args.output_dir}")
    print("="*60)
    
    # 存储所有数据集的结果
    all_similarities_dict = {}
    all_stats = {}
    
    # 对每个数据集运行实验
    for dataset_name in args.datasets:
        try:
            # 加载图数据
            D_inv_A, num_nodes = load_graph_data(dataset_name)
            
            # 运行多次试验
            similarity_matrix, pairwise_similarities, stats = run_multiple_trials(
                D_inv_A, num_nodes, args.feature_dim,
                gamma=args.gamma, num_trials=args.num_trials,
                device=device, dataset_name=dataset_name, output_dir=args.output_dir
            )
            
            # 保存结果
            all_similarities_dict[dataset_name] = pairwise_similarities
            all_stats[dataset_name] = stats
            
            # 绘制热图
            #plot_similarity_heatmap(similarity_matrix, dataset_name, args.output_dir)
            
        except Exception as e:
            print(f"\n  Error processing {dataset_name}: {e}")
            continue
    
    # 绘制综合对比图
    ''' if len(all_similarities_dict) > 0:
            plot_similarity_distribution(all_similarities_dict, args.output_dir)
            plot_pairwise_comparison(all_similarities_dict, args.output_dir)'''
    
    # 保存统计结果到文件
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = f'{args.output_dir}/equilibrium_diversity_stats.txt'
    with open(result_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EQUILIBRIUM DIVERSITY STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Method: Cosine Similarity between equilibria from different initializations\n")
        f.write("Interpretation: Lower similarity indicates more diverse equilibria\n")
        f.write("                (evidence against over-smoothing)\n\n")
        
        f.write(f"Number of trials per dataset: {args.num_trials}\n")
        f.write(f"Number of pairwise comparisons: {args.num_trials * (args.num_trials - 1) // 2}\n\n")
        
        f.write("="*60 + "\n")
        f.write("RESULTS BY DATASET\n")
        f.write("="*60 + "\n\n")
        
        for dataset_name in args.datasets:
            if dataset_name in all_stats:
                stats = all_stats[dataset_name]
                f.write(f"{dataset_name.upper()}:\n")
                f.write(f"  Mean Similarity: {stats['mean']:.6f}\n")
                f.write(f"  Std Deviation:   {stats['std']:.6f}\n")
                f.write(f"  Min Similarity:  {stats['min']:.6f}\n")
                f.write(f"  Max Similarity:  {stats['max']:.6f}\n")
                f.write(f"  Over-smoothing Ratio (>0.95): {stats['over_smooth_ratio']*100:.2f}%\n")
                
                if stats['over_smooth_ratio'] < 0.1:
                    f.write(f"  Conclusion: ✓ NOT over-smoothed (diverse equilibria)\n")
                else:
                    f.write(f"  Conclusion: ✗ Potentially over-smoothed\n")
                f.write("\n")
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {args.output_dir}/")
    print(f"Statistics file: {result_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
