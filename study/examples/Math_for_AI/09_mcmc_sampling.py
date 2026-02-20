"""
MCMC Sampling and Advanced Sampling Techniques

This script demonstrates various sampling methods used in probabilistic ML:
- Rejection sampling
- Importance sampling
- Metropolis-Hastings MCMC
- Reparameterization trick (VAE-style)

These techniques are fundamental for:
- Bayesian inference
- Variational Autoencoders (VAE)
- Generative models
- Monte Carlo estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, multivariate_normal
import torch
import torch.nn.functional as F


def rejection_sampling(target_pdf, proposal_pdf, proposal_sampler, M, n_samples=10000):
    """
    Rejection sampling: sample from target distribution using proposal distribution.

    Args:
        target_pdf: Target probability density function
        proposal_pdf: Proposal probability density function
        proposal_sampler: Function to sample from proposal distribution
        M: Constant such that target_pdf(x) <= M * proposal_pdf(x) for all x
        n_samples: Number of samples to generate

    Returns:
        Array of accepted samples
    """
    samples = []
    n_rejected = 0

    while len(samples) < n_samples:
        # Sample from proposal distribution
        x = proposal_sampler()

        # Sample uniform random variable
        u = np.random.uniform(0, 1)

        # Accept/reject criterion
        acceptance_prob = target_pdf(x) / (M * proposal_pdf(x))

        if u <= acceptance_prob:
            samples.append(x)
        else:
            n_rejected += 1

    acceptance_rate = n_samples / (n_samples + n_rejected)
    print(f"Rejection Sampling - Acceptance Rate: {acceptance_rate:.3f}")

    return np.array(samples)


def importance_sampling(target_pdf, proposal_pdf, proposal_sampler, n_samples=10000):
    """
    Importance sampling: estimate expectations under target distribution
    using samples from proposal distribution.

    Returns samples and importance weights.
    """
    # Sample from proposal
    samples = np.array([proposal_sampler() for _ in range(n_samples)])

    # Compute importance weights
    weights = target_pdf(samples) / proposal_pdf(samples)

    # Normalize weights
    weights = weights / np.sum(weights)

    return samples, weights


def metropolis_hastings(target_pdf, initial_state, proposal_std, n_samples=10000, burn_in=1000):
    """
    Metropolis-Hastings MCMC sampler.

    Uses symmetric Gaussian proposal distribution.

    Args:
        target_pdf: Target probability density (unnormalized is OK)
        initial_state: Starting point for the chain
        proposal_std: Standard deviation of Gaussian proposal
        n_samples: Number of samples to generate
        burn_in: Number of initial samples to discard

    Returns:
        Array of samples, acceptance rate
    """
    samples = []
    current = initial_state
    n_accepted = 0

    for i in range(n_samples + burn_in):
        # Propose new state (symmetric Gaussian)
        proposed = current + np.random.normal(0, proposal_std, size=current.shape)

        # Compute acceptance ratio
        acceptance_ratio = target_pdf(proposed) / target_pdf(current)

        # Accept/reject
        if np.random.uniform(0, 1) < acceptance_ratio:
            current = proposed
            n_accepted += 1

        # Store sample after burn-in
        if i >= burn_in:
            samples.append(current.copy())

    acceptance_rate = n_accepted / (n_samples + burn_in)
    print(f"Metropolis-Hastings - Acceptance Rate: {acceptance_rate:.3f}")

    return np.array(samples), acceptance_rate


def reparameterization_trick_demo():
    """
    Reparameterization trick used in Variational Autoencoders (VAE).

    Instead of sampling z ~ N(mu, sigma^2), we:
    1. Sample epsilon ~ N(0, 1)
    2. Compute z = mu + sigma * epsilon

    This allows gradients to flow through mu and sigma.
    """
    # VAE latent space parameters
    mu = torch.tensor([2.0, -1.0], requires_grad=True)
    log_var = torch.tensor([0.5, 1.0], requires_grad=True)

    # Reparameterization trick
    def sample_latent(mu, log_var, n_samples=1000):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(n_samples, mu.shape[0])
        z = mu + std * eps
        return z

    # Sample latent variables
    z_samples = sample_latent(mu, log_var)

    # Compute a simple loss (e.g., reconstruction + KL divergence)
    # KL divergence for N(mu, sigma^2) and N(0, 1)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Dummy reconstruction loss
    recon_loss = torch.mean(z_samples.pow(2))

    total_loss = recon_loss + 0.1 * kl_loss

    # Backpropagation works because of reparameterization
    total_loss.backward()

    print("\nReparameterization Trick (VAE-style):")
    print(f"Latent mean (mu): {mu.detach().numpy()}")
    print(f"Latent log_var: {log_var.detach().numpy()}")
    print(f"Gradient of mu: {mu.grad.numpy()}")
    print(f"Gradient of log_var: {log_var.grad.numpy()}")
    print(f"KL divergence: {kl_loss.item():.4f}")

    return z_samples.detach().numpy()


def visualize_sampling_comparison():
    """
    Compare different sampling methods on a mixture of Gaussians.
    """
    # Target: mixture of two Gaussians
    def target_pdf(x):
        return 0.6 * norm.pdf(x, loc=-2, scale=0.8) + 0.4 * norm.pdf(x, loc=3, scale=1.2)

    # Proposal: single Gaussian
    proposal_mean = 0.5
    proposal_std = 3.0

    def proposal_pdf(x):
        return norm.pdf(x, loc=proposal_mean, scale=proposal_std)

    def proposal_sampler():
        return np.random.normal(proposal_mean, proposal_std)

    # Find M for rejection sampling
    x_range = np.linspace(-6, 8, 1000)
    M = np.max(target_pdf(x_range) / proposal_pdf(x_range)) * 1.1

    # 1. Rejection Sampling
    print("\n=== Rejection Sampling ===")
    rejection_samples = rejection_sampling(target_pdf, proposal_pdf, proposal_sampler, M, n_samples=5000)

    # 2. Importance Sampling
    print("\n=== Importance Sampling ===")
    importance_samples, weights = importance_sampling(target_pdf, proposal_pdf, proposal_sampler, n_samples=5000)

    # 3. Metropolis-Hastings
    print("\n=== Metropolis-Hastings MCMC ===")
    mh_samples, _ = metropolis_hastings(target_pdf, initial_state=np.array([0.0]),
                                        proposal_std=2.0, n_samples=5000, burn_in=500)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # True distribution
    x = np.linspace(-6, 8, 1000)
    true_pdf = target_pdf(x)

    # Plot 1: Rejection Sampling
    axes[0, 0].hist(rejection_samples, bins=50, density=True, alpha=0.6, label='Samples')
    axes[0, 0].plot(x, true_pdf, 'r-', linewidth=2, label='True PDF')
    axes[0, 0].set_title('Rejection Sampling')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Importance Sampling
    axes[0, 1].hist(importance_samples, bins=50, density=True, alpha=0.3, label='Proposal Samples')
    # Weighted histogram
    axes[0, 1].hist(importance_samples, bins=50, weights=weights*len(weights),
                    density=True, alpha=0.6, label='Weighted Samples')
    axes[0, 1].plot(x, true_pdf, 'r-', linewidth=2, label='True PDF')
    axes[0, 1].set_title('Importance Sampling')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Metropolis-Hastings
    axes[1, 0].hist(mh_samples.flatten(), bins=50, density=True, alpha=0.6, label='MCMC Samples')
    axes[1, 0].plot(x, true_pdf, 'r-', linewidth=2, label='True PDF')
    axes[1, 0].set_title('Metropolis-Hastings MCMC')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: MCMC Trace
    axes[1, 1].plot(mh_samples[:500], alpha=0.7)
    axes[1, 1].set_title('MCMC Trace (first 500 samples)')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Sample Value')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/mcmc_sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /tmp/mcmc_sampling_comparison.png")
    plt.close()


def visualize_reparameterization():
    """
    Visualize samples from reparameterization trick.
    """
    z_samples = reparameterization_trick_demo()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2D scatter
    axes[0].scatter(z_samples[:, 0], z_samples[:, 1], alpha=0.3, s=10)
    axes[0].scatter([2.0], [-1.0], c='red', s=100, marker='x', linewidths=3, label='Mean (μ)')
    axes[0].set_title('Latent Space Samples (Reparameterization Trick)')
    axes[0].set_xlabel('z_1')
    axes[0].set_ylabel('z_2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Marginal distributions
    axes[1].hist(z_samples[:, 0], bins=30, alpha=0.6, label='z_1', density=True)
    axes[1].hist(z_samples[:, 1], bins=30, alpha=0.6, label='z_2', density=True)
    axes[1].set_title('Marginal Distributions')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/reparameterization_trick.png', dpi=150, bbox_inches='tight')
    print("Plot saved to /tmp/reparameterization_trick.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MCMC Sampling and Advanced Sampling Techniques")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run sampling comparisons
    visualize_sampling_comparison()

    # Demonstrate reparameterization trick
    print("\n" + "=" * 60)
    visualize_reparameterization()

    print("\n" + "=" * 60)
    print("Summary:")
    print("- Rejection sampling: Simple but can be inefficient if M is large")
    print("- Importance sampling: Useful for expectation estimation")
    print("- Metropolis-Hastings: MCMC method, eventually converges to target")
    print("- Reparameterization: Enables gradient-based optimization (VAE)")
    print("=" * 60)
