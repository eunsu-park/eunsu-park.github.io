"""
10_nonparametric.py

Demonstrates nonparametric statistical methods:
- Mann-Whitney U test
- Wilcoxon signed-rank test
- Kruskal-Wallis test
- Kolmogorov-Smirnov test
- Kernel density estimation
- Bootstrap confidence intervals
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


def mann_whitney_test():
    """Demonstrate Mann-Whitney U test."""
    print_section("1. Mann-Whitney U Test")

    print("Comparing two independent samples (non-normal distributions)")

    # Generate skewed data
    np.random.seed(42)
    group1 = np.random.exponential(2, 40)
    group2 = np.random.exponential(2.5, 35)

    print(f"\nGroup 1: n={len(group1)}")
    print(f"  Mean: {np.mean(group1):.2f}")
    print(f"  Median: {np.median(group1):.2f}")
    print(f"  Std: {np.std(group1, ddof=1):.2f}")

    print(f"\nGroup 2: n={len(group2)}")
    print(f"  Mean: {np.mean(group2):.2f}")
    print(f"  Median: {np.median(group2):.2f}")
    print(f"  Std: {np.std(group2, ddof=1):.2f}")

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    print(f"\nMann-Whitney U test:")
    print(f"  U statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Significant difference in distributions (p < 0.05)")
    else:
        print(f"  No significant difference (p ≥ 0.05)")

    # Compare with t-test (for comparison)
    t_stat, t_p = stats.ttest_ind(group1, group2)

    print(f"\nFor comparison, t-test p-value: {t_p:.6f}")
    print(f"  (t-test assumes normality; Mann-Whitney does not)")

    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histograms
        axes[0].hist(group1, bins=20, alpha=0.7, label='Group 1', density=True)
        axes[0].hist(group2, bins=20, alpha=0.7, label='Group 2', density=True)
        axes[0].axvline(np.median(group1), color='C0', linestyle='--', linewidth=2)
        axes[0].axvline(np.median(group2), color='C1', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plots
        axes[1].boxplot([group1, group2], labels=['Group 1', 'Group 2'])
        axes[1].set_ylabel('Value')
        axes[1].set_title('Box Plot Comparison')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/mann_whitney.png', dpi=100)
        print("\n[Plot saved to /tmp/mann_whitney.png]")
        plt.close()


def wilcoxon_signed_rank():
    """Demonstrate Wilcoxon signed-rank test."""
    print_section("2. Wilcoxon Signed-Rank Test")

    print("Paired comparison (non-normal differences)")

    # Generate paired data
    np.random.seed(123)
    n = 30
    before = np.random.gamma(3, 2, n)
    treatment_effect = np.random.exponential(1.5, n)
    after = before + treatment_effect

    differences = after - before

    print(f"\nPaired data: n={n}")
    print(f"\nBefore:")
    print(f"  Median: {np.median(before):.2f}")
    print(f"After:")
    print(f"  Median: {np.median(after):.2f}")

    print(f"\nDifferences (after - before):")
    print(f"  Median: {np.median(differences):.2f}")
    print(f"  Mean: {np.mean(differences):.2f}")

    # Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(after, before, alternative='greater')

    print(f"\nWilcoxon signed-rank test (H₁: after > before):")
    print(f"  Statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Significant increase detected (p < 0.05)")
    else:
        print(f"  No significant increase (p ≥ 0.05)")

    # Compare with paired t-test
    t_stat, t_p = stats.ttest_rel(after, before)

    print(f"\nFor comparison, paired t-test p-value: {t_p/2:.6f}")
    print(f"  (one-sided)")

    # Check normality of differences
    shapiro_stat, shapiro_p = stats.shapiro(differences)

    print(f"\nShapiro-Wilk test on differences:")
    print(f"  p-value: {shapiro_p:.6f}")
    if shapiro_p < 0.05:
        print(f"  Differences not normally distributed (Wilcoxon appropriate)")


def kruskal_wallis_test():
    """Demonstrate Kruskal-Wallis test."""
    print_section("3. Kruskal-Wallis Test")

    print("Comparing multiple independent groups (nonparametric ANOVA)")

    # Generate data from different distributions
    np.random.seed(456)
    group1 = np.random.lognormal(1, 0.5, 25)
    group2 = np.random.lognormal(1.2, 0.5, 30)
    group3 = np.random.lognormal(1.4, 0.5, 28)

    print(f"\nGroup 1: n={len(group1)}, median={np.median(group1):.2f}")
    print(f"Group 2: n={len(group2)}, median={np.median(group2):.2f}")
    print(f"Group 3: n={len(group3)}, median={np.median(group3):.2f}")

    # Kruskal-Wallis test
    statistic, p_value = stats.kruskal(group1, group2, group3)

    print(f"\nKruskal-Wallis test:")
    print(f"  H statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Significant differences among groups (p < 0.05)")
    else:
        print(f"  No significant differences (p ≥ 0.05)")

    # Compare with one-way ANOVA
    f_stat, anova_p = stats.f_oneway(group1, group2, group3)

    print(f"\nFor comparison, one-way ANOVA p-value: {anova_p:.6f}")

    if HAS_PLT:
        plt.figure(figsize=(10, 6))
        plt.boxplot([group1, group2, group3],
                   labels=['Group 1', 'Group 2', 'Group 3'])
        plt.ylabel('Value')
        plt.title(f'Kruskal-Wallis Test (H={statistic:.2f}, p={p_value:.4f})')
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/kruskal_wallis.png', dpi=100)
        print("\n[Plot saved to /tmp/kruskal_wallis.png]")
        plt.close()


def kolmogorov_smirnov_test():
    """Demonstrate Kolmogorov-Smirnov test."""
    print_section("4. Kolmogorov-Smirnov Test")

    print("Testing goodness-of-fit to a distribution")

    # Generate data
    np.random.seed(789)
    n = 100

    # Sample 1: Actually normal
    sample_normal = np.random.normal(5, 2, n)

    # Sample 2: Mixture (not normal)
    sample_mixture = np.concatenate([
        np.random.normal(3, 1, n//2),
        np.random.normal(7, 1, n//2)
    ])

    print(f"\nSample size: {n}")

    # Test against normal distribution
    print(f"\n1. Sample from N(5, 2²) tested against N(5, 2²):")

    stat1, p1 = stats.kstest(sample_normal, 'norm',
                             args=(np.mean(sample_normal), np.std(sample_normal, ddof=1)))

    print(f"  KS statistic: {stat1:.4f}")
    print(f"  p-value: {p1:.6f}")
    if p1 > 0.05:
        print(f"  Consistent with normal distribution")

    print(f"\n2. Sample from mixture tested against normal:")

    stat2, p2 = stats.kstest(sample_mixture, 'norm',
                             args=(np.mean(sample_mixture), np.std(sample_mixture, ddof=1)))

    print(f"  KS statistic: {stat2:.4f}")
    print(f"  p-value: {p2:.6f}")
    if p2 < 0.05:
        print(f"  Not consistent with normal distribution (p < 0.05)")

    # Two-sample KS test
    print(f"\n3. Two-sample KS test (comparing two samples):")

    sample_a = np.random.normal(5, 2, 80)
    sample_b = np.random.normal(5.5, 2, 80)

    stat_2s, p_2s = stats.ks_2samp(sample_a, sample_b)

    print(f"  KS statistic: {stat_2s:.4f}")
    print(f"  p-value: {p_2s:.6f}")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Normal sample - histogram
        axes[0, 0].hist(sample_normal, bins=20, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(sample_normal.min(), sample_normal.max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, np.mean(sample_normal), np.std(sample_normal, ddof=1)),
                       'r-', linewidth=2, label='Fitted normal')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title(f'Normal Sample (KS p={p1:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Normal sample - ECDF
        sorted_normal = np.sort(sample_normal)
        ecdf_normal = np.arange(1, len(sorted_normal) + 1) / len(sorted_normal)
        axes[0, 1].plot(sorted_normal, ecdf_normal, 'b-', linewidth=2, label='ECDF')
        axes[0, 1].plot(x, stats.norm.cdf(x, np.mean(sample_normal), np.std(sample_normal, ddof=1)),
                       'r--', linewidth=2, label='Theoretical CDF')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Normal Sample - CDF Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Mixture sample - histogram
        axes[1, 0].hist(sample_mixture, bins=20, density=True, alpha=0.7, edgecolor='black')
        x2 = np.linspace(sample_mixture.min(), sample_mixture.max(), 100)
        axes[1, 0].plot(x2, stats.norm.pdf(x2, np.mean(sample_mixture), np.std(sample_mixture, ddof=1)),
                       'r-', linewidth=2, label='Fitted normal')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'Mixture Sample (KS p={p2:.3f})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Mixture sample - ECDF
        sorted_mixture = np.sort(sample_mixture)
        ecdf_mixture = np.arange(1, len(sorted_mixture) + 1) / len(sorted_mixture)
        axes[1, 1].plot(sorted_mixture, ecdf_mixture, 'b-', linewidth=2, label='ECDF')
        axes[1, 1].plot(x2, stats.norm.cdf(x2, np.mean(sample_mixture), np.std(sample_mixture, ddof=1)),
                       'r--', linewidth=2, label='Theoretical CDF')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Mixture Sample - CDF Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/ks_test.png', dpi=100)
        print("\n[Plot saved to /tmp/ks_test.png]")
        plt.close()


def kernel_density_estimation():
    """Demonstrate kernel density estimation."""
    print_section("5. Kernel Density Estimation")

    # Generate bimodal data
    np.random.seed(999)
    n1, n2 = 100, 80
    mode1 = np.random.normal(2, 0.8, n1)
    mode2 = np.random.normal(6, 1.2, n2)
    data = np.concatenate([mode1, mode2])

    print(f"Generated bimodal data: {len(data)} points")
    print(f"Mode 1: n={n1}, mean={np.mean(mode1):.2f}")
    print(f"Mode 2: n={n2}, mean={np.mean(mode2):.2f}")

    # Kernel density estimation with different bandwidths
    bandwidths = [0.3, 0.5, 1.0, 2.0]

    print(f"\nKernel density estimation with different bandwidths:")

    x_grid = np.linspace(data.min() - 2, data.max() + 2, 500)

    if HAS_PLT:
        plt.figure(figsize=(12, 8))

        for i, bw in enumerate(bandwidths, 1):
            # Gaussian KDE
            kde = stats.gaussian_kde(data, bw_method=bw)
            density = kde(x_grid)

            print(f"\n  Bandwidth = {bw}:")
            print(f"    Peak density: {density.max():.4f}")

            plt.subplot(2, 2, i)
            plt.hist(data, bins=30, density=True, alpha=0.5, edgecolor='black')
            plt.plot(x_grid, density, 'r-', linewidth=2, label=f'KDE (bw={bw})')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'KDE with Bandwidth = {bw}')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/kde.png', dpi=100)
        print("\n[Plot saved to /tmp/kde.png]")
        plt.close()

    # Optimal bandwidth (Scott's rule)
    kde_scott = stats.gaussian_kde(data, bw_method='scott')
    optimal_bw = kde_scott.factor * data.std(ddof=1)

    print(f"\nScott's rule optimal bandwidth: {optimal_bw:.4f}")


def bootstrap_confidence_intervals():
    """Demonstrate bootstrap confidence intervals."""
    print_section("6. Bootstrap Confidence Intervals")

    # Generate skewed data
    np.random.seed(111)
    data = np.random.gamma(2, 2, 80)

    print(f"Sample size: {len(data)}")
    print(f"Sample mean: {np.mean(data):.2f}")
    print(f"Sample median: {np.median(data):.2f}")
    print(f"Sample std: {np.std(data, ddof=1):.2f}")

    # Bootstrap
    n_bootstrap = 5000
    alpha = 0.05

    print(f"\nBootstrap resampling: {n_bootstrap} iterations")

    # Statistics to bootstrap
    boot_means = []
    boot_medians = []
    boot_stds = []
    boot_trimmed_means = []

    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
        boot_medians.append(np.median(boot_sample))
        boot_stds.append(np.std(boot_sample, ddof=1))
        # 10% trimmed mean
        boot_trimmed_means.append(stats.trim_mean(boot_sample, 0.1))

    boot_means = np.array(boot_means)
    boot_medians = np.array(boot_medians)
    boot_stds = np.array(boot_stds)
    boot_trimmed_means = np.array(boot_trimmed_means)

    print(f"\nBootstrap confidence intervals (95%):")

    # Percentile method
    ci_mean = np.percentile(boot_means, [2.5, 97.5])
    ci_median = np.percentile(boot_medians, [2.5, 97.5])
    ci_std = np.percentile(boot_stds, [2.5, 97.5])
    ci_trimmed = np.percentile(boot_trimmed_means, [2.5, 97.5])

    print(f"\n  Mean:")
    print(f"    Point estimate: {np.mean(data):.2f}")
    print(f"    Bootstrap CI: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
    print(f"    Bootstrap SE: {np.std(boot_means, ddof=1):.2f}")

    print(f"\n  Median:")
    print(f"    Point estimate: {np.median(data):.2f}")
    print(f"    Bootstrap CI: [{ci_median[0]:.2f}, {ci_median[1]:.2f}]")

    print(f"\n  Std Dev:")
    print(f"    Point estimate: {np.std(data, ddof=1):.2f}")
    print(f"    Bootstrap CI: [{ci_std[0]:.2f}, {ci_std[1]:.2f}]")

    print(f"\n  Trimmed Mean (10%):")
    print(f"    Point estimate: {stats.trim_mean(data, 0.1):.2f}")
    print(f"    Bootstrap CI: [{ci_trimmed[0]:.2f}, {ci_trimmed[1]:.2f}]")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean
        axes[0, 0].hist(boot_means, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Sample mean')
        axes[0, 0].axvline(ci_mean[0], color='green', linestyle='--', linewidth=1.5)
        axes[0, 0].axvline(ci_mean[1], color='green', linestyle='--', linewidth=1.5, label='95% CI')
        axes[0, 0].set_xlabel('Mean')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Bootstrap Distribution of Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Median
        axes[0, 1].hist(boot_medians, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(np.median(data), color='red', linestyle='--', linewidth=2, label='Sample median')
        axes[0, 1].axvline(ci_median[0], color='green', linestyle='--', linewidth=1.5)
        axes[0, 1].axvline(ci_median[1], color='green', linestyle='--', linewidth=1.5, label='95% CI')
        axes[0, 1].set_xlabel('Median')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Bootstrap Distribution of Median')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Std
        axes[1, 0].hist(boot_stds, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].axvline(np.std(data, ddof=1), color='red', linestyle='--', linewidth=2, label='Sample std')
        axes[1, 0].axvline(ci_std[0], color='blue', linestyle='--', linewidth=1.5)
        axes[1, 0].axvline(ci_std[1], color='blue', linestyle='--', linewidth=1.5, label='95% CI')
        axes[1, 0].set_xlabel('Standard Deviation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Bootstrap Distribution of Std Dev')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Trimmed mean
        axes[1, 1].hist(boot_trimmed_means, bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].axvline(stats.trim_mean(data, 0.1), color='red', linestyle='--', linewidth=2, label='Sample')
        axes[1, 1].axvline(ci_trimmed[0], color='green', linestyle='--', linewidth=1.5)
        axes[1, 1].axvline(ci_trimmed[1], color='green', linestyle='--', linewidth=1.5, label='95% CI')
        axes[1, 1].set_xlabel('Trimmed Mean (10%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Bootstrap Distribution of Trimmed Mean')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/bootstrap_ci.png', dpi=100)
        print("\n[Plot saved to /tmp/bootstrap_ci.png]")
        plt.close()


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  NONPARAMETRIC METHODS DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    mann_whitney_test()
    wilcoxon_signed_rank()
    kruskal_wallis_test()
    kolmogorov_smirnov_test()
    kernel_density_estimation()
    bootstrap_confidence_intervals()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
