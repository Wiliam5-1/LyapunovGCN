#!/usr/bin/env python3
"""
一次性运行4个基线模型（GCN, GAT, APPNP, SGC）的脚本
"""

import subprocess
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_baseline(gpu_id, model_name, args):
    """在指定GPU上运行单个基线模型"""
    cmd = [
        'python', 'grid_search_v6.py',
        '--model', 'new_activation',
        '--dataset', args.dataset,
        '--layers', str(args.layers),
        '--gpus', str(gpu_id),
        '--epsilon', str(args.epsilon),
        '--eta', str(args.eta),
        '--c_activation', str(args.c_activation),
        '--gamma_values', str(args.gamma),
        '--alpha_values', str(args.alpha),
        '--run_baselines', model_name,
        '--use_all_metrics'
    ]
    
    # 添加可选的激活函数类型参数
    if hasattr(args, 'activation_type') and args.activation_type:
        cmd.extend(['--activation_type', args.activation_type])
    
    # 添加可选的三段激活函数参数
    if hasattr(args, 'x1') and args.x1 is not None:
        cmd.extend(['--x1', str(args.x1)])
    
    if hasattr(args, 'x2') and args.x2 is not None:
        cmd.extend(['--x2', str(args.x2)])
    
    print(f"\n{'='*80}")
    print(f"[GPU {gpu_id}] 开始运行 {model_name}")
    print('='*80)

    # 设置环境变量优化显存使用
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

    # 不捕获输出，让其实时显示（与手动运行完全一致）
    result = subprocess.run(cmd, env=env)
    
    print(f"\n{'='*80}")
    if result.returncode == 0:
        print(f"[GPU {gpu_id}] ✅ {model_name} 完成")
        print('='*80)
        return True
    else:
        print(f"[GPU {gpu_id}] ❌ {model_name} 失败 (返回码: {result.returncode})")
        print('='*80)
        return False

def main():
    parser = argparse.ArgumentParser(description='并行运行基线模型脚本')
    
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--layers', type=int, default=64, help='层数')
    parser.add_argument('--gpus', nargs='*', type=int, default=[0, 1, 2, 3, 4], help='GPU ID列表 (默认: 0 1 2 3 4)')
    parser.add_argument('--epsilon', type=float, default=0.15, help='epsilon参数')
    parser.add_argument('--eta', type=float, default=0.0001, help='eta参数')
    parser.add_argument('--c_activation', type=float, default=-0.484754509027, help='c_activation参数')
    parser.add_argument('--gamma', type=float, default=0.4, help='gamma参数')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha参数')
    parser.add_argument('--activation_type', type=str, default='three_segment',
                        choices=['delta_epsilon', 'piecewise_linear', 'three_segment'],
                        help='激活函数类型 (默认: delta_epsilon)')
    parser.add_argument('--x1', type=float, default=0.2, help='三段激活函数的第一个阈值 (默认: 0.0)')
    parser.add_argument('--x2', type=float, default=0.9, help='三段激活函数的第二个阈值 (默认: 1.0)')
    
    args = parser.parse_args()
    
    models = ['GCN', 'GAT', 'APPNP', 'SGC', 'GCNII']
    
    # 检查GPU数量是否匹配模型数量
    if len(args.gpus) != len(models):
        print(f"错误: GPU数量 ({len(args.gpus)}) 必须等于模型数量 ({len(models)})")
        print(f"模型: {models}")
        print(f"GPU: {args.gpus}")
        return

    print("="*80)
    print("并行运行基线模型脚本")
    print("="*80)
    print(f"数据集: {args.dataset}")
    print(f"层数: {args.layers}")
    print(f"GPU分配: {dict(zip(args.gpus, models))}")
    print("="*80)

    # 并行运行
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(run_baseline, gpu_id, model, args): (gpu_id, model)
            for gpu_id, model in zip(args.gpus, models)
        }
        
        success_count = 0
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    print("="*80)
    print(f"完成: {success_count}/{len(models)} 成功")
    print("="*80)

if __name__ == '__main__':
    main()
