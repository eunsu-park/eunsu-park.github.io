"""
01_probability_review.py

Demonstrates fundamental probability concepts:
- Common probability distributions (Normal, Binomial, Poisson, Exponential)
- PDF/CDF plotting
- Central Limit Theorem
- Law of Large Numbers
- Random sampling with numpy/scipy
"""

import numpy as np
from scipy import stats

# Optional plotting
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


def normal_distribution():
    """Demonstrate normal distribution properties."""
    print_section("1. Normal Distribution")

    mu, sigma = 100, 15
    print(f"Normal distribution: μ={mu}, σ={sigma}")

    # Generate samples
    samples = np.random.normal(mu, sigma, 10000)
    print(f"Generated {len(samples)} samples")
    print(f"Sample mean: {np.mean(samples):.2f}")
    print(f"Sample std: {np.std(samples, ddof=1):.2f}")

    # PDF and CDF at specific points
    x_values = [70, 85, 100, 115, 130]
    print("\nPDF and CDF values:")
    for x in x_values:
        pdf = stats.norm.pdf(x, mu, sigma)
        cdf = stats.norm.cdf(x, mu, sigma)
        print(f"  x={x:3d}: PDF={pdf:.6f}, CDF={cdf:.4f}")

    # Percentiles
    percentiles = [0.025, 0.25, 0.50, 0.75, 0.975]
    print("\nPercentiles:")
    for p in percentiles:
        val = stats.norm.ppf(p, mu, sigma)
        print(f"  {p*100:5.1f}%: {val:.2f}")

    if HAS_PLT:
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        pdf = stats.norm.pdf(x, mu, sigma)
        cdf = stats.norm.cdf(x, mu, sigma)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(x, pdf, 'b-', label=f'PDF (μ={mu}, σ={sigma})')
        ax1.fill_between(x, pdf, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Normal Distribution PDF')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(x, cdf, 'r-', label='CDF')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Normal Distribution CDF')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/normal_dist.png', dpi=100)
        print("\n[Plot saved to /tmp/normal_dist.png]")
        plt.close()


def binomial_distribution():
    """Demonstrate binomial distribution."""
    print_section("2. Binomial Distribution")

    n, p = 20, 0.3
    print(f"Binomial distribution: n={n}, p={p}")

    # Theoretical properties
    mean = n * p
    variance = n * p * (1 - p)
    print(f"Theoretical mean: {mean:.2f}")
    print(f"Theoretical variance: {variance:.2f}")

    # Generate samples
    samples = np.random.binomial(n, p, 10000)
    print(f"\nSample mean: {np.mean(samples):.2f}")
    print(f"Sample variance: {np.var(samples, ddof=1):.2f}")

    # PMF for different k values
    print("\nProbability mass function:")
    for k in range(0, n+1, 4):
        pmf = stats.binom.pmf(k, n, p)
        print(f"  P(X={k:2d}) = {pmf:.6f}")

    # CDF
    print("\nCumulative probabilities:")
    for k in [3, 6, 9, 12]:
        cdf = stats.binom.cdf(k, n, p)
        print(f"  P(X≤{k:2d}) = {cdf:.6f}")

    if HAS_PLT:
        k = np.arange(0, n+1)
        pmf = stats.binom.pmf(k, n, p)

        plt.figure(figsize=(10, 5))
        plt.bar(k, pmf, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean={mean:.1f}')
        plt.xlabel('k (number of successes)')
        plt.ylabel('Probability')
        plt.title(f'Binomial Distribution PMF (n={n}, p={p})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/binomial_dist.png', dpi=100)
        print("\n[Plot saved to /tmp/binomial_dist.png]")
        plt.close()


def poisson_exponential():
    """Demonstrate Poisson and Exponential distributions."""
    print_section("3. Poisson and Exponential Distributions")

    # Poisson
    lambda_poisson = 5
    print(f"Poisson distribution: λ={lambda_poisson}")

    samples_poisson = np.random.poisson(lambda_poisson, 10000)
    print(f"Sample mean: {np.mean(samples_poisson):.2f}")
    print(f"Sample variance: {np.var(samples_poisson, ddof=1):.2f}")

    print("\nPoisson PMF:")
    for k in range(0, 11):
        pmf = stats.poisson.pmf(k, lambda_poisson)
        print(f"  P(X={k:2d}) = {pmf:.6f}")

    # Exponential
    lambda_exp = 0.5
    print(f"\nExponential distribution: λ={lambda_exp}")

    samples_exp = np.random.exponential(1/lambda_exp, 10000)
    print(f"Theoretical mean: {1/lambda_exp:.2f}")
    print(f"Sample mean: {np.mean(samples_exp):.2f}")

    print("\nExponential CDF:")
    for x in [0.5, 1.0, 2.0, 3.0, 5.0]:
        cdf = stats.expon.cdf(x, scale=1/lambda_exp)
        print(f"  P(X≤{x:.1f}) = {cdf:.6f}")

    if HAS_PLT:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Poisson
        k = np.arange(0, 15)
        pmf = stats.poisson.pmf(k, lambda_poisson)
        ax1.bar(k, pmf, alpha=0.7, color='purple', edgecolor='black')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Poisson PMF (λ={lambda_poisson})')
        ax1.grid(True, alpha=0.3)

        # Exponential
        x = np.linspace(0, 10, 200)
        pdf = stats.expon.pdf(x, scale=1/lambda_exp)
        ax2.plot(x, pdf, 'orange', linewidth=2)
        ax2.fill_between(x, pdf, alpha=0.3, color='orange')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'Exponential PDF (λ={lambda_exp})')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/poisson_exp_dist.png', dpi=100)
        print("\n[Plot saved to /tmp/poisson_exp_dist.png]")
        plt.close()


def central_limit_theorem():
    """Demonstrate Central Limit Theorem."""
    print_section("4. Central Limit Theorem")

    # Use uniform distribution (clearly non-normal)
    population = np.random.uniform(0, 10, 100000)
    print(f"Population distribution: Uniform(0, 10)")
    print(f"Population mean: {np.mean(population):.2f}")
    print(f"Population std: {np.std(population):.2f}")

    sample_sizes = [2, 5, 10, 30]
    n_samples = 5000

    print(f"\nSampling distribution of means (n_samples={n_samples}):")

    sample_means_dict = {}
    for n in sample_sizes:
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(population, n, replace=True)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)
        sample_means_dict[n] = sample_means

        print(f"\n  Sample size n={n}:")
        print(f"    Mean of sample means: {np.mean(sample_means):.2f}")
        print(f"    Std of sample means: {np.std(sample_means):.2f}")
        print(f"    Theoretical std: {np.std(population)/np.sqrt(n):.2f}")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, n in enumerate(sample_sizes):
            axes[idx].hist(sample_means_dict[n], bins=50, density=True,
                          alpha=0.7, color='skyblue', edgecolor='black')

            # Overlay normal distribution
            mu = np.mean(sample_means_dict[n])
            sigma = np.std(sample_means_dict[n])
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            axes[idx].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                          label='Normal fit')

            axes[idx].set_xlabel('Sample Mean')
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'Sample Size n={n}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/clt_demo.png', dpi=100)
        print("\n[Plot saved to /tmp/clt_demo.png]")
        plt.close()


def law_of_large_numbers():
    """Demonstrate Law of Large Numbers."""
    print_section("5. Law of Large Numbers")

    # Die rolling example
    true_mean = 3.5
    print(f"Rolling a fair die (theoretical mean = {true_mean})")

    n_rolls = 10000
    rolls = np.random.randint(1, 7, n_rolls)

    # Cumulative means
    cumulative_means = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

    # Check convergence at different points
    checkpoints = [10, 100, 1000, 5000, 10000]
    print("\nCumulative mean convergence:")
    for n in checkpoints:
        mean_n = cumulative_means[n-1]
        error = abs(mean_n - true_mean)
        print(f"  n={n:5d}: mean={mean_n:.4f}, error={error:.4f}")

    if HAS_PLT:
        plt.figure(figsize=(12, 5))
        plt.plot(cumulative_means, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(true_mean, color='red', linestyle='--', linewidth=2,
                   label=f'True mean = {true_mean}')
        plt.xlabel('Number of rolls')
        plt.ylabel('Cumulative mean')
        plt.title('Law of Large Numbers: Die Rolling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/lln_demo.png', dpi=100)
        print("\n[Plot saved to /tmp/lln_demo.png]")
        plt.close()


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  PROBABILITY REVIEW DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    normal_distribution()
    binomial_distribution()
    poisson_exponential()
    central_limit_theorem()
    law_of_large_numbers()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
