"""
Test activation function implementations
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import ShiftedDeadZoneActivation, LyapunovSpikingActivation


# ============================================================================
# Configuration - Easy to modify default parameters here
# ============================================================================
# For piecewise_linear activation:
DEFAULT_Y1 = 0.05  # Y-coordinate of first corner point (0.1, y1)
                   # Try: 0.01, 0.02, 0.05, 0.1, etc.
DEFAULT_Y2 = 0.95  # Y-coordinate of second corner point (0.9, y2)
                   # Note: y2 will be auto-adjusted to (1 - y1) to ensure 
                   # the middle segment passes through (0.5, 0.5)
                   # So if you set y1=0.01, y2 will become 0.99

# For delta_epsilon activation:
DEFAULT_EPSILON = 0.3  # Threshold parameter
DEFAULT_C = -1.0       # Sigmoid parameter

# You can either:
# 1. Modify the values above and run: python TEST_ACTIVATION.py --activation piecewise_linear
# 2. Or use command line: python TEST_ACTIVATION.py --activation piecewise_linear --y1 0.01
# ============================================================================


def compute_delta_epsilon(x, epsilon, c):
    """Compute δ_ε(x) function value"""
    delta_x = torch.zeros_like(x)
    
    # Dead zone: x ∈ [0, ε] → 0
    dead_mask = x <= epsilon
    delta_x[dead_mask] = 0
    
    # Active zone: x ∈ (ε, 1]
    active_mask = x > epsilon
    x_active = x[active_mask]
    
    # θ = (x - ε) / (1 - ε)
    theta = (x_active - epsilon) / (1 - epsilon)
    theta = torch.clamp(theta, 0, 1)
    
    # f(θ) = θ * (1 + e^(-c)) / (1 + e^(-cθ))
    numerator = theta * (1 + torch.exp(torch.tensor(-c)))
    denominator = 1 + torch.exp(-c * theta)
    f_theta = numerator / denominator
    
    delta_x[active_mask] = f_theta
    
    return delta_x


def compute_piecewise_linear(x, y1=0.05, y2=0.95):
    """
    Compute piecewise linear activation function
    
    Three segments:
    - [0, 0.1]: Line below y=x, from (0,0) to (0.1, y1)
    - [0.1, 0.9]: Line connecting two corners, passing through (0.5, 0.5)
    - [0.9, 1]: Line above y=x, from (0.9, y2) to (1, 1)
    
    Constraint: For the middle segment to pass through (0.5, 0.5), we need y1 + y2 = 1
    If this constraint is not met, y2 will be automatically adjusted to y2 = 1 - y1
    
    Args:
        x: input tensor
        y1: y-coordinate of first corner point (0.1, y1)
        y2: y-coordinate of second corner point (0.9, y2)
    """
    # Enforce constraint: y1 + y2 = 1 for middle segment to pass through (0.5, 0.5)
    y2 = 1 - y1
    
    y = torch.zeros_like(x)
    
    # Segment 1: [0, 0.1], from (0, 0) to (0.1, y1)
    # y = (y1 / 0.1) * x = 10*y1 * x
    mask1 = x <= 0.1
    y[mask1] = 10 * y1 * x[mask1]
    
    # Segment 2: [0.1, 0.9], from (0.1, y1) to (0.9, y2)
    # Slope: k = (y2 - y1) / (0.9 - 0.1) = (y2 - y1) / 0.8
    # y = y1 + k * (x - 0.1)
    mask2 = (x > 0.1) & (x <= 0.9)
    slope2 = (y2 - y1) / 0.8
    y[mask2] = y1 + slope2 * (x[mask2] - 0.1)
    
    # Segment 3: [0.9, 1], from (0.9, y2) to (1, 1)
    # Slope: k = (1 - y2) / (1 - 0.9) = (1 - y2) / 0.1 = 10*(1-y2)
    # y = y2 + 10*(1-y2) * (x - 0.9)
    mask3 = x > 0.9
    y[mask3] = y2 + 10 * (1 - y2) * (x[mask3] - 0.9)
    
    return y


def compute_area_with_line(x, delta_x, line_y):
    """
    Compute area between δ_ε(x) and y=x
    
    Returns:
        area_above: Area where δ_ε(x) > y=x (above)
        area_below: Area where δ_ε(x) < y=x (below)
    """
    # Calculate difference
    diff = delta_x - line_y
    
    # Compatible with different PyTorch versions
    try:
        trapz = torch.trapezoid  # PyTorch >= 1.11
    except AttributeError:
        trapz = torch.trapz  # PyTorch < 1.11
    
    # Area above: diff > 0
    above_mask = diff > 0
    if above_mask.any():
        area_above = trapz(diff[above_mask], x[above_mask]).item()
    else:
        area_above = 0.0
    
    # Area below: diff < 0 (absolute value)
    below_mask = diff < 0
    if below_mask.any():
        area_below = trapz(-diff[below_mask], x[below_mask]).item()
    else:
        area_below = 0.0
    
    return area_above, area_below


def plot_activation(activation_type='delta_epsilon', epsilon=0.2, c=-1.0, y1=0.05, y2=0.95):
    """
    Plot activation function and compute area with y=x
    
    Args:
        activation_type: 'delta_epsilon' or 'piecewise_linear'
        epsilon: threshold parameter for delta_epsilon
        c: sigmoid parameter for delta_epsilon
        y1: y-coordinate of first corner for piecewise_linear
        y2: y-coordinate of second corner for piecewise_linear (auto-adjusted to 1-y1)
    """
    print("\n" + "="*70)
    if activation_type == 'delta_epsilon':
        print(f"Plotting δ_ε(x) function with ε={epsilon}, c={c}")
    else:
        # For piecewise_linear, enforce constraint y2 = 1 - y1
        y2 = 1 - y1
        print(f"Plotting Piecewise Linear activation function")
        print(f"  Corner points: (0.1, {y1:.4f}) and (0.9, {y2:.4f})")
        print(f"  Constraint: y1 + y2 = {y1 + y2:.4f} (must be 1 for intersection at (0.5, 0.5))")
    print("="*70)
    
    # x-axis (original input)
    x = torch.linspace(0, 1, 2000)
    line_y = x  # Reference line y=x
    
    # Compute activation
    if activation_type == 'delta_epsilon':
        y = compute_delta_epsilon(x, epsilon, c)
        title = f'δ_ε(x) vs y=x (ε={epsilon}, c={c})'
        filename = f'delta_epsilon_eps{epsilon}_c{c}.png'
        label = f'δ_ε(x)'
    else:  # piecewise_linear
        y = compute_piecewise_linear(x, y1, y2)
        title = f'Piecewise Linear Activation vs y=x (y1={y1:.3f}, y2={y2:.3f})'
        filename = f'piecewise_linear_y1{y1:.3f}.png'
        label = 'f(x)'
    
    # Calculate area
    area_above, area_below = compute_area_with_line(x, y, line_y)
    total_area = area_above + area_below
    net_area = area_above - area_below
    
    # Print results
    print("\n" + "-"*70)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-"*70)
    print(f"{'Area Above':<20} {area_above:<15.10f}")
    print(f"{'Area Below':<20} {area_below:<15.10f}")
    print(f"{'Total Area':<20} {total_area:<15.10f}")
    print(f"{'Net Area':<20} {net_area:<15.10f}")
    print("-"*70)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot activation function
    ax.plot(x.numpy(), y.numpy(), linewidth=2.5, label=label, color='blue')
    
    # Plot y=x reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='y=x', zorder=100)
    
    # Add corner markers for piecewise linear
    if activation_type == 'piecewise_linear':
        y2_actual = 1 - y1  # Enforce constraint
        corners = [(0.1, y1), (0.9, y2_actual)]
        for i, (cx, cy) in enumerate(corners):
            ax.plot(cx, cy, 'ro', markersize=8, zorder=200)
            ax.axvline(x=cx, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            # Add text annotation for corners
            ax.text(cx, cy + 0.05, f'({cx:.1f}, {cy:.3f})', 
                   ha='center', fontsize=10, color='red')
        # Mark intersection with y=x
        ax.plot(0.5, 0.5, 'go', markersize=8, zorder=200, label='Intersection (0.5, 0.5)')
        ax.text(0.5, 0.45, '(0.5, 0.5)', ha='center', fontsize=10, color='green')
    
    # Add epsilon line for delta_epsilon
    if activation_type == 'delta_epsilon':
        ax.axvline(x=epsilon, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'ε={epsilon}')
    
    # Grid and styling
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save figure
    save_path = f'results/{filename}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to {save_path}")
    plt.close()
    
    return {
        'area_above': area_above,
        'area_below': area_below,
        'total_area': total_area,
        'net_area': net_area
    }


def main():
    """Run activation function visualization"""
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Activation Function Visualization')
    parser.add_argument('--activation', type=str, default='delta_epsilon',
                        choices=['delta_epsilon', 'piecewise_linear'],
                        help='Activation function type (default: delta_epsilon)')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                        help=f'Threshold parameter for delta_epsilon (default: {DEFAULT_EPSILON})')
    parser.add_argument('--c', type=float, default=DEFAULT_C,
                        help=f'Sigmoid parameter for delta_epsilon (default: {DEFAULT_C})')
    parser.add_argument('--y1', type=float, default=DEFAULT_Y1,
                        help=f'Y-coordinate of first corner (0.1, y1) for piecewise_linear (default: {DEFAULT_Y1})')
    parser.add_argument('--y2', type=float, default=DEFAULT_Y2,
                        help=f'Y-coordinate of second corner (0.9, y2) for piecewise_linear. '
                             f'Note: y2 will be auto-adjusted to 1-y1 to ensure intersection at (0.5, 0.5) (default: {DEFAULT_Y2})')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*70)
    print("Activation Function Visualization")
    print("="*70)
    
    # Plot activation function
    results = plot_activation(
        activation_type=args.activation,
        epsilon=args.epsilon,
        c=args.c,
        y1=args.y1,
        y2=args.y2
    )
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
