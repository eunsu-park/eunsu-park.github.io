"""
Foundation Models - Scaling Laws Implementation

Demonstrates Chinchilla scaling laws and compute-optimal model sizing.
Implements power law relationships between loss, model size, and training data.
Visualizes scaling curves and compute-optimal frontier.

No external dependencies except numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Chinchilla scaling law for loss prediction.

    L(N, D) = A/N^alpha + B/D^beta + E

    Args:
        N: Number of model parameters (non-embedding)
        D: Number of training tokens
        A, B: Scaling coefficients
        alpha, beta: Scaling exponents
        E: Irreducible loss (entropy of natural text)

    Returns:
        Predicted loss value
    """
    return A / (N ** alpha) + B / (D ** beta) + E


def compute_optimal_ratio(A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """
    Compute the compute-optimal ratio N/D from Chinchilla paper.

    At optimum: dL/dN = 0 and dL/dD = 0 under compute constraint.
    Result: N ∝ D^(beta/alpha)

    Returns:
        Optimal ratio coefficient
    """
    # From calculus of Lagrange multipliers with compute constraint
    # Optimal: N = k * D^(beta/alpha)
    ratio_exponent = beta / alpha
    return ratio_exponent


def compute_flops(N, D):
    """
    Estimate FLOPs for training a transformer.

    FLOPs ≈ 6ND (forward + backward pass)

    Args:
        N: Model parameters
        D: Training tokens

    Returns:
        Approximate FLOPs
    """
    return 6 * N * D


def find_optimal_allocation(C, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Given compute budget C (in FLOPs), find optimal N and D.

    Constraint: 6ND = C
    Optimization: minimize L(N, D)

    Solution: N = (C/6)^(beta/(alpha+beta)) * (A*alpha/(B*beta))^(beta/(alpha+beta))

    Args:
        C: Compute budget in FLOPs

    Returns:
        (N_optimal, D_optimal, L_optimal)
    """
    # Analytical solution from Chinchilla paper
    exponent = beta / (alpha + beta)
    coeff = (A * alpha / (B * beta)) ** exponent

    N_opt = coeff * (C / 6) ** exponent
    D_opt = C / (6 * N_opt)
    L_opt = chinchilla_loss(N_opt, D_opt, A, B, alpha, beta, E)

    return N_opt, D_opt, L_opt


def kaplan_scaling_law(N, D, Nc=8.8e13, Dc=5.4e13, alpha_N=0.076, alpha_D=0.095):
    """
    Original Kaplan et al. (2020) scaling law.

    L(N) = (Nc/N)^alpha_N when data is abundant
    L(D) = (Dc/D)^alpha_D when model is large enough

    Returns:
        Predicted loss
    """
    # Use minimum of both constraints
    loss_N = (Nc / N) ** alpha_N
    loss_D = (Dc / D) ** alpha_D
    return max(loss_N, loss_D) + 1.69  # Add irreducible loss


# ============================================================
# Main Demonstrations
# ============================================================

def demo_scaling_curves():
    """Visualize how loss scales with model size and data."""
    print("=" * 60)
    print("DEMO 1: Scaling Curves")
    print("=" * 60)

    # Create range of model sizes (1M to 100B parameters)
    N_range = np.logspace(6, 11, 50)  # 1M to 100B
    D_fixed = 200e9  # 200B tokens (GPT-3 scale)

    losses = [chinchilla_loss(N, D_fixed) for N in N_range]

    print(f"\nFixed data: {D_fixed/1e9:.0f}B tokens")
    print(f"Model size range: {N_range[0]/1e6:.1f}M to {N_range[-1]/1e9:.1f}B params")
    print(f"Loss range: {min(losses):.3f} to {max(losses):.3f}")

    # Show specific points
    for N in [1e9, 7e9, 70e9]:
        loss = chinchilla_loss(N, D_fixed)
        print(f"  {N/1e9:.0f}B params → Loss = {loss:.3f}")

    # Create range of data sizes (1B to 10T tokens)
    D_range = np.logspace(9, 13, 50)  # 1B to 10T
    N_fixed = 7e9  # 7B params (LLaMA-7B scale)

    losses_data = [chinchilla_loss(N_fixed, D) for D in D_range]

    print(f"\nFixed model: {N_fixed/1e9:.0f}B parameters")
    print(f"Data range: {D_range[0]/1e9:.0f}B to {D_range[-1]/1e12:.0f}T tokens")
    print(f"Loss range: {min(losses_data):.3f} to {max(losses_data):.3f}")


def demo_compute_optimal():
    """Find compute-optimal model size for different budgets."""
    print("\n" + "=" * 60)
    print("DEMO 2: Compute-Optimal Allocation")
    print("=" * 60)

    # Different compute budgets (in FLOPs)
    budgets = {
        "GPT-3 (2020)": 3.14e23,      # ~175B params, 300B tokens
        "Chinchilla (2022)": 5.76e23,  # ~70B params, 1.4T tokens
        "LLaMA-65B": 6.3e23,           # ~65B params, 1.4T tokens
        "GPT-4 (estimated)": 1e25,     # Speculation
    }

    print("\nCompute Budget → Optimal Allocation:\n")
    print(f"{'Model':<20} {'FLOPs':<15} {'N (params)':<15} {'D (tokens)':<15} {'Loss':<10}")
    print("-" * 75)

    for name, budget in budgets.items():
        N_opt, D_opt, L_opt = find_optimal_allocation(budget)
        print(f"{name:<20} {budget:.2e}  {N_opt/1e9:>8.1f}B      {D_opt/1e9:>8.0f}B      {L_opt:.3f}")

    # Compare with actual GPT-3 (not optimal by Chinchilla standards)
    print("\n" + "-" * 75)
    print("Comparison: GPT-3 vs Optimal")
    print("-" * 75)

    gpt3_N = 175e9
    gpt3_D = 300e9
    gpt3_flops = compute_flops(gpt3_N, gpt3_D)
    gpt3_loss = chinchilla_loss(gpt3_N, gpt3_D)

    opt_N, opt_D, opt_loss = find_optimal_allocation(gpt3_flops)

    print(f"GPT-3 actual:  {gpt3_N/1e9:.0f}B params, {gpt3_D/1e9:.0f}B tokens → Loss = {gpt3_loss:.3f}")
    print(f"Optimal:       {opt_N/1e9:.0f}B params, {opt_D/1e9:.0f}B tokens → Loss = {opt_loss:.3f}")
    print(f"Improvement:   {gpt3_loss - opt_loss:.3f} reduction in loss")


def demo_scaling_ratio():
    """Demonstrate the compute-optimal N/D ratio."""
    print("\n" + "=" * 60)
    print("DEMO 3: Compute-Optimal Ratio")
    print("=" * 60)

    ratio_exp = compute_optimal_ratio()
    print(f"\nChinchilla optimal ratio: N ∝ D^{ratio_exp:.3f}")
    print(f"This means: D ∝ N^{1/ratio_exp:.3f}")
    print(f"\nRule of thumb: For every 2x increase in model size,")
    print(f"you should increase data by ~{2**(1/ratio_exp):.2f}x")

    print("\n" + "-" * 60)
    print("Scaling trajectory:")
    print("-" * 60)

    base_N = 1e9  # Start with 1B params
    base_D = 20e9  # 20B tokens (Chinchilla optimal for 1B)

    for scale in [1, 2, 4, 8, 16]:
        N = base_N * scale
        D = base_D * (scale ** (1/ratio_exp))
        flops = compute_flops(N, D)
        loss = chinchilla_loss(N, D)

        print(f"{scale:>3}x: {N/1e9:>6.1f}B params, {D/1e9:>7.0f}B tokens, "
              f"{flops:.2e} FLOPs, Loss = {loss:.3f}")


def demo_comparison_with_kaplan():
    """Compare Chinchilla vs Kaplan scaling laws."""
    print("\n" + "=" * 60)
    print("DEMO 4: Chinchilla vs Kaplan Scaling Laws")
    print("=" * 60)

    test_models = [
        (1e9, "1B"),
        (7e9, "7B"),
        (13e9, "13B"),
        (70e9, "70B"),
    ]

    D = 200e9  # 200B tokens

    print(f"\nWith {D/1e9:.0f}B training tokens:\n")
    print(f"{'Size':<10} {'Chinchilla':<15} {'Kaplan':<15} {'Difference':<10}")
    print("-" * 50)

    for N, name in test_models:
        chin_loss = chinchilla_loss(N, D)
        kaplan_loss = kaplan_scaling_law(N, D)
        diff = kaplan_loss - chin_loss

        print(f"{name:<10} {chin_loss:.4f}          {kaplan_loss:.4f}          {diff:+.4f}")

    print("\nNote: Kaplan (2020) predicted more aggressive scaling benefits.")
    print("Chinchilla (2022) revised this with better data efficiency emphasis.")


def plot_scaling_laws():
    """Generate comprehensive scaling law visualizations."""
    print("\n" + "=" * 60)
    print("DEMO 5: Visualization (plots generated)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss vs Model Size (fixed data)
    ax1 = axes[0, 0]
    N_range = np.logspace(6, 11, 100)
    D_fixed = 200e9
    losses = [chinchilla_loss(N, D_fixed) for N in N_range]

    ax1.loglog(N_range, losses, 'b-', linewidth=2, label='Chinchilla')
    ax1.set_xlabel('Model Parameters (N)', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title(f'Loss vs Model Size (D = {D_fixed/1e9:.0f}B tokens)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Loss vs Training Data (fixed model)
    ax2 = axes[0, 1]
    D_range = np.logspace(9, 13, 100)
    N_fixed = 7e9
    losses_data = [chinchilla_loss(N_fixed, D) for D in D_range]

    ax2.loglog(D_range, losses_data, 'r-', linewidth=2, label='Chinchilla')
    ax2.set_xlabel('Training Tokens (D)', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title(f'Loss vs Training Data (N = {N_fixed/1e9:.0f}B params)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Compute-Optimal Frontier
    ax3 = axes[1, 0]
    compute_budgets = np.logspace(21, 25, 50)
    N_opts = []
    D_opts = []

    for C in compute_budgets:
        N_opt, D_opt, _ = find_optimal_allocation(C)
        N_opts.append(N_opt)
        D_opts.append(D_opt)

    ax3.loglog(N_opts, D_opts, 'g-', linewidth=2, label='Optimal frontier')

    # Add specific model points
    models = {
        'GPT-3': (175e9, 300e9),
        'Chinchilla': (70e9, 1400e9),
        'LLaMA-65B': (65e9, 1400e9),
    }

    for name, (n, d) in models.items():
        ax3.plot(n, d, 'o', markersize=8, label=name)

    ax3.set_xlabel('Model Parameters (N)', fontsize=11)
    ax3.set_ylabel('Training Tokens (D)', fontsize=11)
    ax3.set_title('Compute-Optimal Frontier', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Loss Landscape (2D)
    ax4 = axes[1, 1]
    N_grid = np.logspace(9, 11, 30)
    D_grid = np.logspace(10, 13, 30)
    N_mesh, D_mesh = np.meshgrid(N_grid, D_grid)

    L_mesh = np.zeros_like(N_mesh)
    for i in range(len(D_grid)):
        for j in range(len(N_grid)):
            L_mesh[i, j] = chinchilla_loss(N_mesh[i, j], D_mesh[i, j])

    contour = ax4.contour(N_mesh, D_mesh, L_mesh, levels=15, cmap='viridis')
    ax4.clabel(contour, inline=True, fontsize=8)
    ax4.set_xlabel('Model Parameters (N)', fontsize=11)
    ax4.set_ylabel('Training Tokens (D)', fontsize=11)
    ax4.set_title('Loss Landscape L(N, D)', fontsize=12)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Foundation_Models/scaling_laws.png', dpi=150)
    print("\nPlot saved to: scaling_laws.png")
    print("Shows: (1) Loss vs N, (2) Loss vs D, (3) Optimal frontier, (4) Loss landscape")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Foundation Models: Scaling Laws")
    print("=" * 60)

    demo_scaling_curves()
    demo_compute_optimal()
    demo_scaling_ratio()
    demo_comparison_with_kaplan()
    plot_scaling_laws()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Loss scales as power laws: L ~ N^(-α) and L ~ D^(-β)")
    print("2. Chinchilla law: For compute budget C, optimal N ∝ C^0.45, D ∝ C^0.55")
    print("3. Most models are overtrained (too many params, too little data)")
    print("4. Doubling model size requires ~2.4x more data for optimality")
    print("=" * 60)
