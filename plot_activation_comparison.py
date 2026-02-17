import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib参数
matplotlib.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})

# 颜色调色板 - 参考grid_search_v6.py的配色方案
_COLOR_PALETTE = {
    'piecewise': '#4198b9',      # 深青蓝色
    'three_segment': '#6bb3c0',  # 中等青绿色
    'identity': '#2c3e50'        # 深灰色用于参考线
}

# 参数设置 - 方便修改
# 左图参数 (piecewise_linear)
epsilon = 0.2
y1 = 0.05

# 右图参数 (three_segment)
x1 = 0.2
x2 = 0.9

# 共同参数
c = -0.484754509027

def piecewise_linear_activation(x, y1, epsilon):
    """
    分段线性激活函数 - 与 models_v6.py 保持一致

    三个段落:
    - [0, epsilon]: 从 (0,0) 到 (epsilon, y1) 的直线
    - [epsilon, 1-epsilon]: 连接两个拐点的直线，经过 (0.5, 0.5)
    - [1-epsilon, 1]: 从 (1-epsilon, y2) 到 (1, 1) 的直线

    约束: y1 + y2 = 1 以确保中间段经过 (0.5, 0.5)
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

def three_segment_activation(x, x1, x2, c):
    """
    三段激活函数 - 从models_v6.py中提取
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

# 生成输入数据
x = torch.linspace(0, 1, 1000)

# 计算激活函数值
y_piecewise = piecewise_linear_activation(x, y1, epsilon)
y_three_segment = three_segment_activation(x, x1, x2, c)

# 创建子图 - 使用与grid_search_v6.py类似的布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

# ==================== 左图：Piecewise Linear激活函数 ====================
# 绘制激活函数曲线
ax1.plot(x.numpy(), y_piecewise.numpy(),
         color=_COLOR_PALETTE['piecewise'],
         label='Piecewise Linear',
         linewidth=2.8,
         alpha=0.95,
         zorder=3)

# 绘制参考线 (identity function)
ax1.plot(x.numpy(), x.numpy(),
         color=_COLOR_PALETTE['identity'],
         linestyle='--',
         linewidth=1.8,
         alpha=0.8,
         label='y=x',
         zorder=2)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r'$x$', fontsize=13, fontweight='bold', color='#2c3e50')
ax1.set_ylabel(r'$\delta_\varepsilon(x)$', fontsize=13, fontweight='bold', color='#2c3e50')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, color='gray')
ax1.set_axisbelow(True)

# 设置轴线样式
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#2c3e50')

# 设置图例
legend = ax1.legend(fontsize=11, loc='best', framealpha=0.95,
                   edgecolor='#2c3e50', fancybox=True, shadow=True)
legend.set_zorder(10)
ax1.set_facecolor('#f9f9f9')

# ==================== 右图：Three Segment激活函数 ====================
# 绘制激活函数曲线
ax2.plot(x.numpy(), y_three_segment.numpy(),
         color=_COLOR_PALETTE['three_segment'],
         label='SDZA',
         linewidth=2.8,
         alpha=0.95,
         zorder=3)

# 绘制参考线 (identity function)
ax2.plot(x.numpy(), x.numpy(),
         color=_COLOR_PALETTE['identity'],
         linestyle='--',
         linewidth=1.8,
         alpha=0.8,
         label='Identity',
         zorder=2)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel(r'$x$', fontsize=13, fontweight='bold', color='#2c3e50')
ax2.set_ylabel(r'$\delta_\varepsilon(x)$', fontsize=13, fontweight='bold', color='#2c3e50')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, color='gray')
ax2.set_axisbelow(True)

# 设置轴线样式
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#2c3e50')

# 设置图例
legend = ax2.legend(fontsize=11, loc='best', framealpha=0.95,
                   edgecolor='#2c3e50', fancybox=True, shadow=True)
legend.set_zorder(10)
ax2.set_facecolor('#f9f9f9')

# 调整布局并保存
plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()