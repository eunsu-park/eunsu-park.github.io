"""
02_sampling_estimation.py

Demonstrates sampling methods and estimation techniques:
- Simple random sampling
- Stratified sampling
- Bootstrap estimation
- Confidence intervals
- Bias and variance of estimators
- Maximum Likelihood Estimation (MLE)
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


def simple_random_sampling():
    """Demonstrate simple random sampling."""
    print_section("1. Simple Random Sampling")

    # Create population
    population_size = 10000
    population = np.random.normal(100, 15, population_size)

    print(f"Population size: {population_size}")
    print(f"Population mean: {np.mean(population):.2f}")
    print(f"Population std: {np.std(population, ddof=0):.2f}")

    # Different sample sizes
    sample_sizes = [10, 50, 100, 500]
    n_replications = 1000

    print(f"\nSimple random sampling ({n_replications} replications):")

    for n in sample_sizes:
        sample_means = []
        for _ in range(n_replications):
            sample = np.random.choice(population, n, replace=False)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        print(f"\n  Sample size n={n}:")
        print(f"    Mean of estimates: {np.mean(sample_means):.2f}")
        print(f"    Std of estimates: {np.std(sample_means, ddof=1):.2f}")
        print(f"    Theoretical SE: {np.std(population)/np.sqrt(n):.2f}")
        print(f"    Min estimate: {np.min(sample_means):.2f}")
        print(f"    Max estimate: {np.max(sample_means):.2f}")


def stratified_sampling():
    """Demonstrate stratified sampling."""
    print_section("2. Stratified Sampling")

    # Create stratified population
    # Stratum 1: 60%, mean=90
    # Stratum 2: 30%, mean=110
    # Stratum 3: 10%, mean=130

    n1, n2, n3 = 6000, 3000, 1000
    stratum1 = np.random.normal(90, 12, n1)
    stratum2 = np.random.normal(110, 15, n2)
    stratum3 = np.random.normal(130, 18, n3)

    population = np.concatenate([stratum1, stratum2, stratum3])

    print("Population composition:")
    print(f"  Stratum 1: n={n1}, mean={np.mean(stratum1):.2f}")
    print(f"  Stratum 2: n={n2}, mean={np.mean(stratum2):.2f}")
    print(f"  Stratum 3: n={n3}, mean={np.mean(stratum3):.2f}")
    print(f"  Overall mean: {np.mean(population):.2f}")

    # Compare simple vs stratified sampling
    total_sample_size = 300
    n_replications = 1000

    simple_means = []
    stratified_means = []

    for _ in range(n_replications):
        # Simple random sampling
        simple_sample = np.random.choice(population, total_sample_size, replace=False)
        simple_means.append(np.mean(simple_sample))

        # Stratified sampling (proportional allocation)
        s1_sample = np.random.choice(stratum1, 180, replace=False)  # 60%
        s2_sample = np.random.choice(stratum2, 90, replace=False)   # 30%
        s3_sample = np.random.choice(stratum3, 30, replace=False)   # 10%
        stratified_sample = np.concatenate([s1_sample, s2_sample, s3_sample])
        stratified_means.append(np.mean(stratified_sample))

    simple_means = np.array(simple_means)
    stratified_means = np.array(stratified_means)

    print(f"\nComparison (n={total_sample_size}, {n_replications} replications):")
    print(f"\n  Simple Random Sampling:")
    print(f"    Mean: {np.mean(simple_means):.2f}")
    print(f"    Std: {np.std(simple_means, ddof=1):.2f}")

    print(f"\n  Stratified Sampling:")
    print(f"    Mean: {np.mean(stratified_means):.2f}")
    print(f"    Std: {np.std(stratified_means, ddof=1):.2f}")

    efficiency = np.var(simple_means, ddof=1) / np.var(stratified_means, ddof=1)
    print(f"\n  Efficiency gain: {efficiency:.2f}x")
    print(f"  Variance reduction: {(1 - 1/efficiency)*100:.1f}%")


def bootstrap_estimation():
    """Demonstrate bootstrap estimation."""
    print_section("3. Bootstrap Estimation")

    # Original sample
    np.random.seed(42)
    sample = np.random.lognormal(3, 0.5, 100)

    print(f"Original sample size: {len(sample)}")
    print(f"Sample mean: {np.mean(sample):.2f}")
    print(f"Sample median: {np.median(sample):.2f}")
    print(f"Sample std: {np.std(sample, ddof=1):.2f}")

    # Bootstrap
    n_bootstrap = 5000
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_stds = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample, len(sample), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
        bootstrap_medians.append(np.median(bootstrap_sample))
        bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))

    bootstrap_means = np.array(bootstrap_means)
    bootstrap_medians = np.array(bootstrap_medians)
    bootstrap_stds = np.array(bootstrap_stds)

    print(f"\nBootstrap results ({n_bootstrap} resamples):")

    print(f"\n  Mean:")
    print(f"    Bootstrap mean: {np.mean(bootstrap_means):.2f}")
    print(f"    Bootstrap SE: {np.std(bootstrap_means, ddof=1):.2f}")
    print(f"    95% CI: [{np.percentile(bootstrap_means, 2.5):.2f}, "
          f"{np.percentile(bootstrap_means, 97.5):.2f}]")

    print(f"\n  Median:")
    print(f"    Bootstrap mean: {np.mean(bootstrap_medians):.2f}")
    print(f"    Bootstrap SE: {np.std(bootstrap_medians, ddof=1):.2f}")
    print(f"    95% CI: [{np.percentile(bootstrap_medians, 2.5):.2f}, "
          f"{np.percentile(bootstrap_medians, 97.5):.2f}]")

    if HAS_PLT:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].hist(bootstrap_means, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(sample), color='red', linestyle='--', label='Original')
        axes[0].set_xlabel('Mean')
        axes[0].set_title('Bootstrap Distribution of Mean')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(bootstrap_medians, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[1].axvline(np.median(sample), color='red', linestyle='--', label='Original')
        axes[1].set_xlabel('Median')
        axes[1].set_title('Bootstrap Distribution of Median')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(bootstrap_stds, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[2].axvline(np.std(sample, ddof=1), color='red', linestyle='--', label='Original')
        axes[2].set_xlabel('Std Dev')
        axes[2].set_title('Bootstrap Distribution of Std Dev')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/bootstrap_demo.png', dpi=100)
        print("\n[Plot saved to /tmp/bootstrap_demo.png]")
        plt.close()


def confidence_intervals():
    """Demonstrate confidence interval construction."""
    print_section("4. Confidence Intervals")

    # True population
    true_mean = 100
    true_std = 15
    sample_size = 50
    n_samples = 1000
    confidence_level = 0.95

    print(f"Population: N({true_mean}, {true_std}²)")
    print(f"Sample size: {sample_size}")
    print(f"Confidence level: {confidence_level*100}%")
    print(f"Number of samples: {n_samples}")

    # Generate samples and compute CIs
    coverage_count = 0
    ci_widths = []

    for _ in range(n_samples):
        sample = np.random.normal(true_mean, true_std, sample_size)
        sample_mean = np.mean(sample)
        sample_se = stats.sem(sample)

        # t-based CI
        ci = stats.t.interval(confidence_level, sample_size - 1,
                             loc=sample_mean, scale=sample_se)
        ci_widths.append(ci[1] - ci[0])

        if ci[0] <= true_mean <= ci[1]:
            coverage_count += 1

    coverage_rate = coverage_count / n_samples

    print(f"\nResults:")
    print(f"  Coverage rate: {coverage_rate*100:.1f}%")
    print(f"  Expected coverage: {confidence_level*100}%")
    print(f"  Average CI width: {np.mean(ci_widths):.2f}")
    print(f"  CI width std: {np.std(ci_widths, ddof=1):.2f}")

    # Example single sample CI
    example_sample = np.random.normal(true_mean, true_std, sample_size)
    ex_mean = np.mean(example_sample)
    ex_se = stats.sem(example_sample)
    ex_ci = stats.t.interval(confidence_level, sample_size - 1,
                            loc=ex_mean, scale=ex_se)

    print(f"\nExample single sample:")
    print(f"  Sample mean: {ex_mean:.2f}")
    print(f"  Standard error: {ex_se:.2f}")
    print(f"  95% CI: [{ex_ci[0]:.2f}, {ex_ci[1]:.2f}]")
    print(f"  Contains true mean: {ex_ci[0] <= true_mean <= ex_ci[1]}")


def bias_variance_estimators():
    """Demonstrate bias and variance of estimators."""
    print_section("5. Bias and Variance of Estimators")

    # True population variance
    true_var = 225  # std = 15
    sample_size = 20
    n_samples = 5000

    print(f"True population variance: {true_var}")
    print(f"Sample size: {sample_size}")
    print(f"Number of samples: {n_samples}")

    # Two estimators for variance
    biased_vars = []
    unbiased_vars = []

    for _ in range(n_samples):
        sample = np.random.normal(100, np.sqrt(true_var), sample_size)

        # Biased estimator (divide by n)
        biased_var = np.var(sample, ddof=0)
        biased_vars.append(biased_var)

        # Unbiased estimator (divide by n-1)
        unbiased_var = np.var(sample, ddof=1)
        unbiased_vars.append(unbiased_var)

    biased_vars = np.array(biased_vars)
    unbiased_vars = np.array(unbiased_vars)

    print("\nBiased estimator (divide by n):")
    print(f"  Mean: {np.mean(biased_vars):.2f}")
    print(f"  Bias: {np.mean(biased_vars) - true_var:.2f}")
    print(f"  Variance: {np.var(biased_vars, ddof=1):.2f}")
    print(f"  MSE: {np.mean((biased_vars - true_var)**2):.2f}")

    print("\nUnbiased estimator (divide by n-1):")
    print(f"  Mean: {np.mean(unbiased_vars):.2f}")
    print(f"  Bias: {np.mean(unbiased_vars) - true_var:.2f}")
    print(f"  Variance: {np.var(unbiased_vars, ddof=1):.2f}")
    print(f"  MSE: {np.mean((unbiased_vars - true_var)**2):.2f}")


def mle_normal():
    """Demonstrate Maximum Likelihood Estimation for normal distribution."""
    print_section("6. MLE for Normal Distribution")

    # True parameters
    true_mu = 50
    true_sigma = 10
    sample_size = 100

    # Generate sample
    np.random.seed(123)
    sample = np.random.normal(true_mu, true_sigma, sample_size)

    print(f"True parameters: μ={true_mu}, σ={true_sigma}")
    print(f"Sample size: {sample_size}")

    # MLE estimates
    mle_mu = np.mean(sample)
    mle_sigma = np.std(sample, ddof=0)  # MLE uses n, not n-1

    print(f"\nMLE estimates:")
    print(f"  μ̂ = {mle_mu:.4f}")
    print(f"  σ̂ = {mle_sigma:.4f}")

    # Unbiased estimate for comparison
    unbiased_sigma = np.std(sample, ddof=1)
    print(f"\nUnbiased σ estimate: {unbiased_sigma:.4f}")

    # Log-likelihood
    log_likelihood = np.sum(stats.norm.logpdf(sample, mle_mu, mle_sigma))
    print(f"\nLog-likelihood at MLE: {log_likelihood:.2f}")

    # Verify MLE is maximum
    print("\nLog-likelihood for different parameter values:")
    mu_candidates = [mle_mu - 2, mle_mu - 1, mle_mu, mle_mu + 1, mle_mu + 2]
    for mu_cand in mu_candidates:
        ll = np.sum(stats.norm.logpdf(sample, mu_cand, mle_sigma))
        marker = " <-- MLE" if abs(mu_cand - mle_mu) < 0.01 else ""
        print(f"  μ={mu_cand:6.2f}: LL={ll:.2f}{marker}")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  SAMPLING AND ESTIMATION DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    simple_random_sampling()
    stratified_sampling()
    bootstrap_estimation()
    confidence_intervals()
    bias_variance_estimators()
    mle_normal()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
