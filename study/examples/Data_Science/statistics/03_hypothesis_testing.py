"""
03_hypothesis_testing.py

Demonstrates hypothesis testing methods:
- t-tests (one-sample, two-sample, paired)
- Chi-square tests
- ANOVA (one-way)
- p-value computation
- Statistical power analysis
- Multiple testing correction (Bonferroni, FDR)
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


def one_sample_t_test():
    """Demonstrate one-sample t-test."""
    print_section("1. One-Sample t-Test")

    # Known population mean
    mu_0 = 100

    # Generate sample with different true mean
    np.random.seed(42)
    sample = np.random.normal(105, 15, 50)

    print(f"Null hypothesis: μ = {mu_0}")
    print(f"Alternative hypothesis: μ ≠ {mu_0}")
    print(f"\nSample size: {len(sample)}")
    print(f"Sample mean: {np.mean(sample):.2f}")
    print(f"Sample std: {np.std(sample, ddof=1):.2f}")

    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(sample, mu_0)

    print(f"\nTest results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  degrees of freedom: {len(sample) - 1}")

    alpha = 0.05
    print(f"\nDecision (α={alpha}):")
    if p_value < alpha:
        print(f"  Reject H₀ (p={p_value:.6f} < {alpha})")
        print(f"  Evidence suggests μ ≠ {mu_0}")
    else:
        print(f"  Fail to reject H₀ (p={p_value:.6f} ≥ {alpha})")
        print(f"  Insufficient evidence against μ = {mu_0}")

    # Confidence interval
    ci = stats.t.interval(0.95, len(sample)-1,
                         loc=np.mean(sample),
                         scale=stats.sem(sample))
    print(f"\n95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")


def two_sample_t_test():
    """Demonstrate two-sample t-test."""
    print_section("2. Two-Sample t-Test")

    # Generate two samples
    np.random.seed(123)
    group1 = np.random.normal(100, 15, 40)
    group2 = np.random.normal(108, 15, 45)

    print("Group 1:")
    print(f"  n = {len(group1)}")
    print(f"  mean = {np.mean(group1):.2f}")
    print(f"  std = {np.std(group1, ddof=1):.2f}")

    print("\nGroup 2:")
    print(f"  n = {len(group2)}")
    print(f"  mean = {np.mean(group2):.2f}")
    print(f"  std = {np.std(group2, ddof=1):.2f}")

    # Independent samples t-test (equal variance assumed)
    t_stat, p_value = stats.ttest_ind(group1, group2)

    print(f"\nIndependent samples t-test (equal variance):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    # Welch's t-test (unequal variance)
    t_stat_welch, p_value_welch = stats.ttest_ind(group1, group2, equal_var=False)

    print(f"\nWelch's t-test (unequal variance):")
    print(f"  t-statistic: {t_stat_welch:.4f}")
    print(f"  p-value: {p_value_welch:.6f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")


def paired_t_test():
    """Demonstrate paired t-test."""
    print_section("3. Paired t-Test")

    # Generate paired data (before/after)
    np.random.seed(456)
    n = 30
    before = np.random.normal(120, 20, n)
    treatment_effect = np.random.normal(10, 5, n)
    after = before + treatment_effect

    print(f"Sample size: {n} pairs")
    print(f"\nBefore treatment:")
    print(f"  Mean: {np.mean(before):.2f}")
    print(f"  Std: {np.std(before, ddof=1):.2f}")

    print(f"\nAfter treatment:")
    print(f"  Mean: {np.mean(after):.2f}")
    print(f"  Std: {np.std(after, ddof=1):.2f}")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(after, before)

    differences = after - before
    print(f"\nDifferences (after - before):")
    print(f"  Mean: {np.mean(differences):.2f}")
    print(f"  Std: {np.std(differences, ddof=1):.2f}")

    print(f"\nPaired t-test results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"\n  Significant difference detected (p < 0.05)")
    else:
        print(f"\n  No significant difference (p ≥ 0.05)")


def chi_square_test():
    """Demonstrate chi-square test."""
    print_section("4. Chi-Square Test")

    # Goodness of fit test
    print("Chi-square goodness of fit test:")
    print("Testing if a die is fair\n")

    observed = np.array([48, 52, 45, 53, 47, 55])
    expected = np.array([50, 50, 50, 50, 50, 50])

    print("Observed frequencies:", observed)
    print("Expected frequencies:", expected)

    chi2_stat, p_value = stats.chisquare(observed, expected)

    print(f"\nχ² statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Degrees of freedom: {len(observed) - 1}")

    if p_value > 0.05:
        print(f"\nNo evidence against fairness (p = {p_value:.4f})")

    # Contingency table test
    print("\n" + "-" * 70)
    print("Chi-square test of independence:")
    print("Testing relationship between treatment and outcome\n")

    # Contingency table: rows=treatment, columns=outcome
    contingency_table = np.array([
        [20, 30],  # Treatment A: success, failure
        [35, 15]   # Treatment B: success, failure
    ])

    print("Contingency table:")
    print("                Success  Failure")
    print(f"Treatment A:       {contingency_table[0,0]}       {contingency_table[0,1]}")
    print(f"Treatment B:       {contingency_table[1,0]}       {contingency_table[1,1]}")

    chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)

    print(f"\nχ² statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")

    print("\nExpected frequencies:")
    print(expected_freq)

    if p_value < 0.05:
        print(f"\nSignificant association detected (p = {p_value:.4f})")


def one_way_anova():
    """Demonstrate one-way ANOVA."""
    print_section("5. One-Way ANOVA")

    # Generate three groups with different means
    np.random.seed(789)
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(105, 15, 30)
    group3 = np.random.normal(112, 15, 30)

    print("Group statistics:")
    for i, group in enumerate([group1, group2, group3], 1):
        print(f"\n  Group {i}:")
        print(f"    n = {len(group)}")
        print(f"    mean = {np.mean(group):.2f}")
        print(f"    std = {np.std(group, ddof=1):.2f}")

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(group1, group2, group3)

    print(f"\nOne-way ANOVA results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"\n  Significant differences among groups (p < 0.05)")
        print(f"  At least one group mean differs from others")
    else:
        print(f"\n  No significant differences (p ≥ 0.05)")

    # Calculate effect size (eta-squared)
    all_data = np.concatenate([group1, group2, group3])
    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2
                    for g in [group1, group2, group3])
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total

    print(f"\nEffect size (η²): {eta_squared:.4f}")


def power_analysis():
    """Demonstrate statistical power analysis."""
    print_section("6. Statistical Power Analysis")

    # Parameters
    effect_size = 0.5  # Cohen's d
    alpha = 0.05
    sample_sizes = [10, 20, 30, 50, 100, 200]

    print(f"Effect size (Cohen's d): {effect_size}")
    print(f"Significance level (α): {alpha}")
    print(f"\nPower analysis for two-sample t-test:")

    # Simulate power for different sample sizes
    print(f"\n  {'n':>5} {'Power':>8} {'Description':>15}")
    print(f"  {'-'*5} {'-'*8} {'-'*15}")

    for n in sample_sizes:
        # Monte Carlo estimation of power
        n_simulations = 5000
        rejections = 0

        for _ in range(n_simulations):
            # Generate data with true effect
            group1 = np.random.normal(0, 1, n)
            group2 = np.random.normal(effect_size, 1, n)

            _, p_value = stats.ttest_ind(group1, group2)
            if p_value < alpha:
                rejections += 1

        power = rejections / n_simulations

        # Power interpretation
        if power < 0.5:
            desc = "Underpowered"
        elif power < 0.8:
            desc = "Moderate"
        elif power < 0.95:
            desc = "Good"
        else:
            desc = "Excellent"

        print(f"  {n:5d} {power:8.3f} {desc:>15}")


def multiple_testing_correction():
    """Demonstrate multiple testing correction."""
    print_section("7. Multiple Testing Correction")

    # Simulate multiple hypothesis tests
    np.random.seed(999)
    n_tests = 20
    alpha = 0.05

    print(f"Number of tests: {n_tests}")
    print(f"Nominal α: {alpha}")

    # Generate p-values (15 null true, 5 alternative true)
    p_values = []

    # Null true (no effect)
    for _ in range(15):
        sample1 = np.random.normal(0, 1, 30)
        sample2 = np.random.normal(0, 1, 30)
        _, p = stats.ttest_ind(sample1, sample2)
        p_values.append(p)

    # Alternative true (real effect)
    for _ in range(5):
        sample1 = np.random.normal(0, 1, 30)
        sample2 = np.random.normal(0.8, 1, 30)
        _, p = stats.ttest_ind(sample1, sample2)
        p_values.append(p)

    p_values = np.array(p_values)

    print(f"\nUncorrected results:")
    print(f"  Significant tests (p < {alpha}): {np.sum(p_values < alpha)}")
    print(f"  Smallest p-value: {np.min(p_values):.6f}")
    print(f"  Largest p-value: {np.max(p_values):.6f}")

    # Bonferroni correction
    bonferroni_alpha = alpha / n_tests
    bonferroni_rejections = np.sum(p_values < bonferroni_alpha)

    print(f"\nBonferroni correction:")
    print(f"  Adjusted α: {bonferroni_alpha:.6f}")
    print(f"  Significant tests: {bonferroni_rejections}")

    # Benjamini-Hochberg (FDR)
    sorted_p = np.sort(p_values)
    sorted_indices = np.argsort(p_values)

    # Find largest i where p(i) <= (i/n) * alpha
    fdr_threshold = 0
    fdr_rejections = 0

    for i in range(n_tests):
        threshold = ((i + 1) / n_tests) * alpha
        if sorted_p[i] <= threshold:
            fdr_threshold = sorted_p[i]
            fdr_rejections = i + 1

    print(f"\nBenjamini-Hochberg (FDR) correction:")
    print(f"  FDR level: {alpha}")
    print(f"  Threshold p-value: {fdr_threshold:.6f}")
    print(f"  Significant tests: {fdr_rejections}")

    print(f"\nSummary:")
    print(f"  No correction: {np.sum(p_values < alpha)} rejections")
    print(f"  Bonferroni: {bonferroni_rejections} rejections (conservative)")
    print(f"  FDR: {fdr_rejections} rejections (balanced)")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  HYPOTHESIS TESTING DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    one_sample_t_test()
    two_sample_t_test()
    paired_t_test()
    chi_square_test()
    one_way_anova()
    power_analysis()
    multiple_testing_correction()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
