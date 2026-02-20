"""
Probability Distributions and Statistical Inference

This script demonstrates:
1. Common probability distributions (Gaussian, Bernoulli, Poisson, Exponential)
2. Maximum Likelihood Estimation (MLE) for Gaussian parameters
3. Maximum A Posteriori (MAP) estimation with Gaussian prior
4. Bayesian update visualization

Author: Math for AI Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple


def plot_common_distributions():
    """Visualize common probability distributions used in ML."""
    print("\n" + "="*60)
    print("1. Common Probability Distributions")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Gaussian (Normal) Distribution
    ax = axes[0, 0]
    x = np.linspace(-5, 5, 1000)

    for mu, sigma in [(0, 0.5), (0, 1), (0, 2), (2, 1)]:
        pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, linewidth=2, label=f'μ={mu}, σ={sigma}')

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Gaussian Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    print("Gaussian: f(x|μ,σ²) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))")
    print("  Use: Continuous data, errors, neural network activations")

    # 2. Bernoulli Distribution
    ax = axes[0, 1]
    x = np.array([0, 1])

    for p in [0.2, 0.5, 0.8]:
        pmf = np.array([1-p, p])
        ax.bar(x + (p-0.5)*0.15, pmf, width=0.12, alpha=0.7, label=f'p={p}')

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Probability Mass', fontsize=11)
    ax.set_title('Bernoulli Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    print("\nBernoulli: P(X=1) = p, P(X=0) = 1-p")
    print("  Use: Binary classification, coin flips")

    # 3. Poisson Distribution
    ax = axes[1, 0]
    x = np.arange(0, 20)

    for lambda_ in [1, 4, 10]:
        pmf = stats.poisson.pmf(x, lambda_)
        ax.plot(x, pmf, 'o-', linewidth=2, markersize=5, label=f'λ={lambda_}')

    ax.set_xlabel('k (count)', fontsize=11)
    ax.set_ylabel('Probability Mass', fontsize=11)
    ax.set_title('Poisson Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    print("\nPoisson: P(X=k) = (λ^k * e^(-λ)) / k!")
    print("  Use: Count data, rare events, arrivals per time period")

    # 4. Exponential Distribution
    ax = axes[1, 1]
    x = np.linspace(0, 5, 1000)

    for lambda_ in [0.5, 1, 2]:
        pdf = stats.expon.pdf(x, scale=1/lambda_)
        ax.plot(x, pdf, linewidth=2, label=f'λ={lambda_}')

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Exponential Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    print("\nExponential: f(x|λ) = λ * e^(-λx) for x >= 0")
    print("  Use: Time between events, survival analysis, waiting times")

    plt.tight_layout()
    plt.savefig('probability_distributions.png', dpi=150, bbox_inches='tight')
    print("\nSaved distributions plot to 'probability_distributions.png'")


def mle_gaussian_demo():
    """
    Maximum Likelihood Estimation for Gaussian parameters.

    Given data X = {x₁, ..., xₙ} from N(μ, σ²), find MLE for μ and σ².

    Likelihood: L(μ,σ²) = ∏ᵢ (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))
    Log-likelihood: ℓ(μ,σ²) = -n/2 log(2π) - n/2 log(σ²) - Σ(xᵢ-μ)²/(2σ²)

    MLE solutions:
    μ̂ = (1/n) Σxᵢ  (sample mean)
    σ̂² = (1/n) Σ(xᵢ-μ̂)²  (sample variance)
    """
    print("\n" + "="*60)
    print("2. Maximum Likelihood Estimation (MLE)")
    print("="*60)

    # True parameters
    true_mu = 2.0
    true_sigma = 1.5

    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    data = np.random.normal(true_mu, true_sigma, n_samples)

    # MLE estimates
    mle_mu = np.mean(data)
    mle_sigma = np.std(data, ddof=0)  # ddof=0 for MLE (biased estimator)

    print(f"\nTrue parameters: μ = {true_mu}, σ = {true_sigma}")
    print(f"MLE estimates:   μ̂ = {mle_mu:.4f}, σ̂ = {mle_sigma:.4f}")
    print(f"Sample size: n = {n_samples}")

    # Plot log-likelihood surface
    mu_range = np.linspace(0, 4, 100)
    sigma_range = np.linspace(0.5, 3, 100)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)

    def log_likelihood(mu, sigma, data):
        """Compute log-likelihood for Gaussian."""
        n = len(data)
        ll = -n/2 * np.log(2*np.pi) - n * np.log(sigma)
        ll -= np.sum((data - mu)**2) / (2 * sigma**2)
        return ll

    # Compute log-likelihood for grid
    LL = np.zeros_like(MU)
    for i in range(MU.shape[0]):
        for j in range(MU.shape[1]):
            LL[i, j] = log_likelihood(MU[i, j], SIGMA[i, j], data)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Data histogram with MLE fit
    ax1.hist(data, bins=20, density=True, alpha=0.6, color='skyblue',
             edgecolor='black', label='Data')

    x_plot = np.linspace(data.min(), data.max(), 200)
    ax1.plot(x_plot, stats.norm.pdf(x_plot, mle_mu, mle_sigma),
             'r-', linewidth=3, label=f'MLE Fit: N({mle_mu:.2f}, {mle_sigma:.2f}²)')
    ax1.plot(x_plot, stats.norm.pdf(x_plot, true_mu, true_sigma),
             'g--', linewidth=2, label=f'True: N({true_mu}, {true_sigma}²)')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('MLE Fit to Data', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log-likelihood contour
    contour = ax2.contour(MU, SIGMA, LL, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)

    ax2.plot(mle_mu, mle_sigma, 'r*', markersize=20, label='MLE Estimate')
    ax2.plot(true_mu, true_sigma, 'go', markersize=12, label='True Parameters')

    ax2.set_xlabel('μ', fontsize=12)
    ax2.set_ylabel('σ', fontsize=12)
    ax2.set_title('Log-Likelihood Surface', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mle_gaussian.png', dpi=150, bbox_inches='tight')
    print("\nSaved MLE visualization to 'mle_gaussian.png'")


def map_estimation_demo():
    """
    Maximum A Posteriori (MAP) estimation with Gaussian prior.

    Prior: μ ~ N(μ₀, σ₀²)
    Likelihood: x | μ ~ N(μ, σ²)
    Posterior: μ | x ~ N(μₙ, σₙ²)

    where:
    μₙ = (σ²μ₀ + nσ₀²x̄) / (σ² + nσ₀²)
    σₙ² = (σ²σ₀²) / (σ² + nσ₀²)
    """
    print("\n" + "="*60)
    print("3. Maximum A Posteriori (MAP) Estimation")
    print("="*60)

    # True mean
    true_mu = 5.0
    data_sigma = 1.0

    # Prior parameters
    prior_mu = 3.0
    prior_sigma = 2.0

    print(f"\nPrior belief: μ ~ N({prior_mu}, {prior_sigma}²)")
    print(f"Data distribution: X ~ N(μ, {data_sigma}²)")
    print(f"True mean: μ = {true_mu}")

    # Generate increasing amounts of data
    np.random.seed(42)
    sample_sizes = [1, 5, 20, 100]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, n in enumerate(sample_sizes):
        ax = axes[idx]

        # Generate data
        data = np.random.normal(true_mu, data_sigma, n)
        data_mean = np.mean(data)

        # MLE (just the sample mean)
        mle_mu = data_mean

        # MAP estimate (posterior mean)
        posterior_mu = (data_sigma**2 * prior_mu + n * prior_sigma**2 * data_mean) / \
                      (data_sigma**2 + n * prior_sigma**2)
        posterior_sigma = np.sqrt((data_sigma**2 * prior_sigma**2) /
                                  (data_sigma**2 + n * prior_sigma**2))

        print(f"\nn = {n} samples:")
        print(f"  Sample mean: {data_mean:.4f}")
        print(f"  MLE:  μ̂ = {mle_mu:.4f}")
        print(f"  MAP:  μ̂ = {posterior_mu:.4f}")
        print(f"  Posterior: N({posterior_mu:.4f}, {posterior_sigma:.4f}²)")

        # Plot prior, likelihood, and posterior
        mu_range = np.linspace(-2, 10, 500)

        # Prior
        prior_pdf = stats.norm.pdf(mu_range, prior_mu, prior_sigma)
        ax.plot(mu_range, prior_pdf, 'b--', linewidth=2, label='Prior')

        # Likelihood (as function of μ)
        likelihood_sigma = data_sigma / np.sqrt(n)
        likelihood_pdf = stats.norm.pdf(mu_range, data_mean, likelihood_sigma)
        likelihood_pdf = likelihood_pdf / likelihood_pdf.max() * prior_pdf.max()  # Scale for visualization
        ax.plot(mu_range, likelihood_pdf, 'g:', linewidth=2, label='Likelihood (scaled)')

        # Posterior
        posterior_pdf = stats.norm.pdf(mu_range, posterior_mu, posterior_sigma)
        ax.plot(mu_range, posterior_pdf, 'r-', linewidth=2, label='Posterior')

        # Mark estimates
        ax.axvline(prior_mu, color='b', linestyle='--', alpha=0.5, label='Prior mean')
        ax.axvline(mle_mu, color='g', linestyle=':', alpha=0.5, label='MLE')
        ax.axvline(posterior_mu, color='r', linestyle='-', alpha=0.5, label='MAP')
        ax.axvline(true_mu, color='k', linestyle='-', linewidth=2, label='True μ')

        ax.set_xlabel('μ', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'n = {n} samples', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('map_estimation.png', dpi=150, bbox_inches='tight')
    print("\nSaved MAP estimation plot to 'map_estimation.png'")


def bayesian_update_demo():
    """Visualize sequential Bayesian updates."""
    print("\n" + "="*60)
    print("4. Sequential Bayesian Update")
    print("="*60)

    # Setup
    true_mu = 5.0
    data_sigma = 1.0
    prior_mu = 2.0
    prior_sigma = 3.0

    np.random.seed(42)
    n_updates = 5
    data_points = np.random.normal(true_mu, data_sigma, n_updates)

    print(f"Prior: μ ~ N({prior_mu}, {prior_sigma}²)")
    print(f"True mean: {true_mu}")
    print(f"\nSequential observations: {data_points}")

    fig, ax = plt.subplots(figsize=(12, 7))
    mu_range = np.linspace(-5, 12, 500)

    # Plot prior
    current_mu = prior_mu
    current_sigma = prior_sigma
    pdf = stats.norm.pdf(mu_range, current_mu, current_sigma)
    ax.plot(mu_range, pdf, linewidth=3, label=f'Prior: N({current_mu:.2f}, {current_sigma:.2f}²)',
            color='blue')

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_updates))

    # Sequential updates
    for i, x in enumerate(data_points):
        # Update: posterior becomes new prior
        new_mu = (data_sigma**2 * current_mu + current_sigma**2 * x) / \
                 (data_sigma**2 + current_sigma**2)
        new_sigma = np.sqrt((data_sigma**2 * current_sigma**2) /
                           (data_sigma**2 + current_sigma**2))

        pdf = stats.norm.pdf(mu_range, new_mu, new_sigma)
        ax.plot(mu_range, pdf, linewidth=2.5,
                label=f'After x_{i+1}={x:.2f}: N({new_mu:.2f}, {new_sigma:.2f}²)',
                color=colors[i])

        current_mu = new_mu
        current_sigma = new_sigma

        print(f"\nUpdate {i+1}: observed x = {x:.4f}")
        print(f"  Posterior: N({new_mu:.4f}, {new_sigma:.4f}²)")

    # Mark true value
    ax.axvline(true_mu, color='black', linestyle='--', linewidth=2,
               label=f'True μ = {true_mu}')

    ax.set_xlabel('μ', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('Sequential Bayesian Update', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bayesian_update.png', dpi=150, bbox_inches='tight')
    print("\nSaved Bayesian update plot to 'bayesian_update.png'")
    print(f"\nFinal posterior: N({current_mu:.4f}, {current_sigma:.4f}²)")
    print(f"Distance from true mean: {abs(current_mu - true_mu):.4f}")


if __name__ == "__main__":
    print("="*60)
    print("Probability Distributions and Statistical Inference")
    print("="*60)

    # Run demonstrations
    plot_common_distributions()
    mle_gaussian_demo()
    map_estimation_demo()
    bayesian_update_demo()

    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Gaussian: Most common distribution, Central Limit Theorem")
    print("2. MLE: Find parameters that maximize likelihood of observed data")
    print("3. MAP: Incorporates prior knowledge, balances prior and likelihood")
    print("4. Bayesian update: Posterior from step n becomes prior for step n+1")
    print("5. More data → posterior concentrates around true value")
