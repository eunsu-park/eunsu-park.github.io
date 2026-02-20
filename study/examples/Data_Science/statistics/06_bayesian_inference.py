"""
06_bayesian_inference.py

Demonstrates advanced Bayesian inference methods:
- MCMC basics (Metropolis-Hastings)
- Gibbs sampling simple example
- Posterior sampling
- Convergence diagnostics (trace plots)
- Bayesian linear regression
"""

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("matplotlib not available; skipping plots\n")


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def metropolis_hastings_normal():
    """Demonstrate Metropolis-Hastings for sampling from normal distribution."""
    print_section("1. Metropolis-Hastings Algorithm")

    print("Sampling from N(10, 2²) using Metropolis-Hastings")

    # Target distribution
    target_mean = 10
    target_std = 2

    def log_target(x):
        """Log of target distribution."""
        return stats.norm.logpdf(x, target_mean, target_std)

    # MCMC parameters
    n_samples = 10000
    proposal_std = 3
    burn_in = 1000

    # Initialize
    current = 0  # Starting point
    samples = []
    accepted = 0

    print(f"\nMCMC parameters:")
    print(f"  Iterations: {n_samples}")
    print(f"  Burn-in: {burn_in}")
    print(f"  Proposal std: {proposal_std}")

    # Run Metropolis-Hastings
    for i in range(n_samples):
        # Propose new state
        proposed = current + np.random.normal(0, proposal_std)

        # Acceptance ratio
        log_ratio = log_target(proposed) - log_target(current)
        accept_prob = min(1, np.exp(log_ratio))

        # Accept or reject
        if np.random.uniform() < accept_prob:
            current = proposed
            accepted += 1

        samples.append(current)

    samples = np.array(samples)
    samples_after_burnin = samples[burn_in:]

    print(f"\nResults:")
    print(f"  Acceptance rate: {accepted/n_samples:.3f}")
    print(f"  Sample mean: {np.mean(samples_after_burnin):.3f} (true: {target_mean})")
    print(f"  Sample std: {np.std(samples_after_burnin, ddof=1):.3f} (true: {target_std})")

    # Effective sample size (simple autocorrelation-based estimate)
    autocorr_lag1 = np.corrcoef(samples_after_burnin[:-1], samples_after_burnin[1:])[0, 1]
    ess_approx = len(samples_after_burnin) * (1 - autocorr_lag1) / (1 + autocorr_lag1)

    print(f"  Autocorrelation (lag 1): {autocorr_lag1:.3f}")
    print(f"  Approx. effective sample size: {ess_approx:.0f}")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Trace plot
        axes[0, 0].plot(samples, alpha=0.7, linewidth=0.5)
        axes[0, 0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
        axes[0, 0].axhline(target_mean, color='green', linestyle='--', label='True mean')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Trace Plot')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram
        axes[0, 1].hist(samples_after_burnin, bins=50, density=True,
                       alpha=0.7, edgecolor='black', label='MCMC samples')
        x = np.linspace(target_mean - 4*target_std, target_mean + 4*target_std, 200)
        axes[0, 1].plot(x, stats.norm.pdf(x, target_mean, target_std),
                       'r-', linewidth=2, label='True distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Posterior Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Running mean
        running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        axes[1, 0].plot(running_mean, alpha=0.7)
        axes[1, 0].axhline(target_mean, color='red', linestyle='--', label='True mean')
        axes[1, 0].axvline(burn_in, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Running Mean')
        axes[1, 0].set_title('Convergence of Mean')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Autocorrelation
        max_lag = 50
        autocorr = [np.corrcoef(samples_after_burnin[:-lag], samples_after_burnin[lag:])[0, 1]
                   if lag > 0 else 1.0 for lag in range(max_lag)]
        axes[1, 1].bar(range(max_lag), autocorr, alpha=0.7)
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/mcmc_metropolis.png', dpi=100)
        print("\n[Plot saved to /tmp/mcmc_metropolis.png]")
        plt.close()


def gibbs_sampling_bivariate():
    """Demonstrate Gibbs sampling for bivariate normal."""
    print_section("2. Gibbs Sampling")

    print("Sampling from bivariate normal using Gibbs sampling")

    # Target: bivariate normal with correlation
    mu = np.array([3, 5])
    rho = 0.7
    sigma_x = 1.5
    sigma_y = 2.0
    cov_matrix = np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])

    print(f"\nTarget distribution:")
    print(f"  μ = {mu}")
    print(f"  ρ = {rho}")
    print(f"  σ_x = {sigma_x}, σ_y = {sigma_y}")

    # Conditional distributions
    def sample_x_given_y(y):
        """Sample x | y from conditional normal."""
        mu_cond = mu[0] + rho * (sigma_x / sigma_y) * (y - mu[1])
        sigma_cond = sigma_x * np.sqrt(1 - rho**2)
        return np.random.normal(mu_cond, sigma_cond)

    def sample_y_given_x(x):
        """Sample y | x from conditional normal."""
        mu_cond = mu[1] + rho * (sigma_y / sigma_x) * (x - mu[0])
        sigma_cond = sigma_y * np.sqrt(1 - rho**2)
        return np.random.normal(mu_cond, sigma_cond)

    # Gibbs sampling
    n_samples = 5000
    burn_in = 500

    x_samples = np.zeros(n_samples)
    y_samples = np.zeros(n_samples)

    # Initialize
    x_samples[0] = 0
    y_samples[0] = 0

    print(f"\nGibbs sampling:")
    print(f"  Iterations: {n_samples}")
    print(f"  Burn-in: {burn_in}")

    # Run Gibbs
    for i in range(1, n_samples):
        x_samples[i] = sample_x_given_y(y_samples[i-1])
        y_samples[i] = sample_y_given_x(x_samples[i])

    # Remove burn-in
    x_final = x_samples[burn_in:]
    y_final = y_samples[burn_in:]

    print(f"\nResults (after burn-in):")
    print(f"  Mean x: {np.mean(x_final):.3f} (true: {mu[0]})")
    print(f"  Mean y: {np.mean(y_final):.3f} (true: {mu[1]})")
    print(f"  Std x: {np.std(x_final, ddof=1):.3f} (true: {sigma_x})")
    print(f"  Std y: {np.std(y_final, ddof=1):.3f} (true: {sigma_y})")
    print(f"  Correlation: {np.corrcoef(x_final, y_final)[0,1]:.3f} (true: {rho})")

    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Trace plots
        axes[0].plot(x_samples, alpha=0.5, label='x')
        axes[0].plot(y_samples, alpha=0.5, label='y')
        axes[0].axvline(burn_in, color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(mu[0], color='blue', linestyle='--', alpha=0.3)
        axes[0].axhline(mu[1], color='orange', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Trace Plots')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Joint distribution
        axes[1].scatter(x_final, y_final, alpha=0.3, s=5)
        axes[1].scatter([mu[0]], [mu[1]], color='red', s=100, marker='x',
                       label='True mean', zorder=5)

        # Add ellipse for true distribution
        from matplotlib.patches import Ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence ellipse
        ellipse = Ellipse(mu, width, height, angle=angle,
                         facecolor='none', edgecolor='red', linewidth=2, label='95% ellipse')
        axes[1].add_patch(ellipse)

        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Joint Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')

        plt.tight_layout()
        plt.savefig('/tmp/gibbs_sampling.png', dpi=100)
        print("\n[Plot saved to /tmp/gibbs_sampling.png]")
        plt.close()


def bayesian_linear_regression():
    """Demonstrate Bayesian linear regression."""
    print_section("3. Bayesian Linear Regression")

    # Generate data
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 10, n)
    true_beta_0 = 5
    true_beta_1 = 2
    sigma_true = 2
    y = true_beta_0 + true_beta_1 * x + np.random.normal(0, sigma_true, n)

    print(f"True model: y = {true_beta_0} + {true_beta_1}*x + N(0, {sigma_true}²)")
    print(f"Sample size: {n}")

    # Design matrix
    X = np.column_stack([np.ones(n), x])

    # Prior for coefficients: N(0, 100*I) - weakly informative
    prior_mean = np.zeros(2)
    prior_cov = 100 * np.eye(2)

    # Likelihood precision (assume known for simplicity)
    tau = 1 / sigma_true**2

    # Posterior (closed form for normal prior, normal likelihood)
    # Posterior covariance
    post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + tau * (X.T @ X))

    # Posterior mean
    post_mean = post_cov @ (np.linalg.inv(prior_cov) @ prior_mean + tau * (X.T @ y))

    print(f"\nPosterior distribution of coefficients:")
    print(f"  β₀: mean={post_mean[0]:.3f}, std={np.sqrt(post_cov[0,0]):.3f}")
    print(f"  β₁: mean={post_mean[1]:.3f}, std={np.sqrt(post_cov[1,1]):.3f}")
    print(f"  Correlation: {post_cov[0,1] / np.sqrt(post_cov[0,0] * post_cov[1,1]):.3f}")

    # Sample from posterior
    n_posterior_samples = 2000
    beta_samples = np.random.multivariate_normal(post_mean, post_cov, n_posterior_samples)

    print(f"\nPosterior samples (n={n_posterior_samples}):")
    print(f"  β₀: mean={np.mean(beta_samples[:,0]):.3f}")
    print(f"  β₁: mean={np.mean(beta_samples[:,1]):.3f}")

    # Credible intervals
    beta_0_ci = np.percentile(beta_samples[:,0], [2.5, 97.5])
    beta_1_ci = np.percentile(beta_samples[:,1], [2.5, 97.5])

    print(f"\n95% Credible intervals:")
    print(f"  β₀: [{beta_0_ci[0]:.3f}, {beta_0_ci[1]:.3f}]")
    print(f"  β₁: [{beta_1_ci[0]:.3f}, {beta_1_ci[1]:.3f}]")

    # Prediction with uncertainty
    x_new = np.linspace(0, 10, 100)
    X_new = np.column_stack([np.ones(len(x_new)), x_new])

    # Posterior predictive samples
    y_pred_samples = []
    for beta in beta_samples[:500]:  # Use subset for speed
        y_pred_mean = X_new @ beta
        y_pred = y_pred_mean + np.random.normal(0, sigma_true, len(x_new))
        y_pred_samples.append(y_pred)

    y_pred_samples = np.array(y_pred_samples)
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Posterior samples of coefficients
        axes[0].scatter(beta_samples[:,0], beta_samples[:,1],
                       alpha=0.3, s=5, label='Posterior samples')
        axes[0].scatter([true_beta_0], [true_beta_1], color='red',
                       s=100, marker='*', label='True values', zorder=5)
        axes[0].scatter([post_mean[0]], [post_mean[1]], color='green',
                       s=100, marker='x', label='Posterior mean', zorder=5)
        axes[0].set_xlabel('β₀ (intercept)')
        axes[0].set_ylabel('β₁ (slope)')
        axes[0].set_title('Posterior Distribution of Coefficients')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Predictions
        axes[1].scatter(x, y, alpha=0.6, s=30, label='Data')
        axes[1].plot(x_new, y_pred_mean, 'r-', linewidth=2, label='Posterior mean')
        axes[1].fill_between(x_new, y_pred_lower, y_pred_upper,
                            alpha=0.3, color='red', label='95% Prediction interval')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Bayesian Linear Regression Predictions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/bayesian_regression.png', dpi=100)
        print("\n[Plot saved to /tmp/bayesian_regression.png]")
        plt.close()


def convergence_diagnostics():
    """Demonstrate MCMC convergence diagnostics."""
    print_section("4. Convergence Diagnostics")

    print("Multiple chains for convergence assessment")

    # Target: N(5, 1.5²)
    target_mean = 5
    target_std = 1.5

    def log_target(x):
        return stats.norm.logpdf(x, target_mean, target_std)

    # Run multiple chains
    n_chains = 4
    n_samples = 3000
    burn_in = 500
    proposal_std = 2

    chains = []

    print(f"\nRunning {n_chains} chains:")
    print(f"  Samples per chain: {n_samples}")
    print(f"  Burn-in: {burn_in}")

    # Different starting points
    starting_points = [-5, 0, 10, 15]

    for chain_id, start in enumerate(starting_points):
        current = start
        samples = []

        for i in range(n_samples):
            proposed = current + np.random.normal(0, proposal_std)
            log_ratio = log_target(proposed) - log_target(current)

            if np.random.uniform() < min(1, np.exp(log_ratio)):
                current = proposed

            samples.append(current)

        chains.append(np.array(samples))
        print(f"  Chain {chain_id+1}: start={start:5.1f}, mean={np.mean(samples[burn_in:]):.3f}")

    # Gelman-Rubin diagnostic (simple version)
    chains_after_burnin = [c[burn_in:] for c in chains]

    # Within-chain variance
    W = np.mean([np.var(c, ddof=1) for c in chains_after_burnin])

    # Between-chain variance
    chain_means = [np.mean(c) for c in chains_after_burnin]
    B = np.var(chain_means, ddof=1) * len(chains_after_burnin[0])

    # R-hat
    var_plus = ((len(chains_after_burnin[0]) - 1) * W + B) / len(chains_after_burnin[0])
    R_hat = np.sqrt(var_plus / W)

    print(f"\nGelman-Rubin R̂ statistic: {R_hat:.4f}")
    if R_hat < 1.1:
        print(f"  Chains have converged (R̂ < 1.1)")
    else:
        print(f"  Chains may not have converged (R̂ ≥ 1.1)")

    if HAS_PLT:
        plt.figure(figsize=(12, 5))

        for i, chain in enumerate(chains):
            plt.plot(chain, alpha=0.7, label=f'Chain {i+1}')

        plt.axvline(burn_in, color='red', linestyle='--', alpha=0.5, label='Burn-in end')
        plt.axhline(target_mean, color='black', linestyle='--', alpha=0.5, label='True mean')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Multiple Chains (R̂={R_hat:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/convergence_chains.png', dpi=100)
        print("\n[Plot saved to /tmp/convergence_chains.png]")
        plt.close()


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  BAYESIAN INFERENCE DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    metropolis_hastings_normal()
    gibbs_sampling_bivariate()
    bayesian_linear_regression()
    convergence_diagnostics()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
