"""
05_bayesian_basics.py

Demonstrates basic Bayesian inference concepts:
- Bayes' theorem examples
- Conjugate priors (Beta-Binomial, Normal-Normal)
- Posterior computation
- Credible intervals
- Comparison with frequentist approach
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


def bayes_theorem_discrete():
    """Demonstrate Bayes' theorem with discrete example."""
    print_section("1. Bayes' Theorem - Discrete Example")

    # Medical test example
    print("Medical diagnostic test scenario:")
    print("  Disease prevalence: 1%")
    print("  Test sensitivity (P(+|Disease)): 95%")
    print("  Test specificity (P(-|No disease)): 90%")

    # Prior
    p_disease = 0.01
    p_no_disease = 0.99

    # Likelihood
    p_pos_given_disease = 0.95
    p_pos_given_no_disease = 0.10

    # Marginal probability of positive test
    p_pos = (p_pos_given_disease * p_disease +
             p_pos_given_no_disease * p_no_disease)

    # Posterior using Bayes' theorem
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

    print(f"\nQuestion: If test is positive, what's P(Disease|+)?")
    print(f"\nCalculation:")
    print(f"  P(+) = P(+|D)P(D) + P(+|¬D)P(¬D)")
    print(f"       = {p_pos_given_disease}×{p_disease} + {p_pos_given_no_disease}×{p_no_disease}")
    print(f"       = {p_pos:.4f}")
    print(f"\n  P(D|+) = P(+|D)P(D) / P(+)")
    print(f"         = ({p_pos_given_disease}×{p_disease}) / {p_pos:.4f}")
    print(f"         = {p_disease_given_pos:.4f}")

    print(f"\nResult: Only {p_disease_given_pos*100:.2f}% chance of disease despite positive test!")
    print(f"Reason: Low prevalence (strong prior)")


def beta_binomial_conjugate():
    """Demonstrate Beta-Binomial conjugate prior."""
    print_section("2. Beta-Binomial Conjugate Prior")

    print("Estimating coin bias θ (probability of heads)")

    # Prior: Beta(α, β)
    alpha_prior = 2
    beta_prior = 2
    print(f"\nPrior: Beta({alpha_prior}, {beta_prior})")
    print(f"  Prior mean: {alpha_prior/(alpha_prior + beta_prior):.3f}")
    print(f"  Prior expresses weak belief in fairness")

    # Data: observed coin flips
    n_heads = 7
    n_tails = 3
    n_total = n_heads + n_tails

    print(f"\nObserved data: {n_heads} heads, {n_tails} tails in {n_total} flips")

    # Posterior: Beta(α + n_heads, β + n_tails)
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails

    print(f"\nPosterior: Beta({alpha_post}, {beta_post})")
    post_mean = alpha_post / (alpha_post + beta_post)
    post_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))

    print(f"  Posterior mean: {post_mean:.3f}")
    print(f"  Posterior std: {np.sqrt(post_var):.3f}")

    # Credible interval
    credible_interval = stats.beta.interval(0.95, alpha_post, beta_post)
    print(f"  95% Credible Interval: [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")

    # MLE for comparison
    mle = n_heads / n_total
    print(f"\nFrequentist MLE: {mle:.3f}")
    print(f"Bayesian posterior mean: {post_mean:.3f}")
    print(f"  (Bayesian estimate pulled toward prior)")

    if HAS_PLT:
        theta = np.linspace(0, 1, 200)
        prior_pdf = stats.beta.pdf(theta, alpha_prior, beta_prior)
        likelihood = stats.beta.pdf(theta, n_heads + 1, n_tails + 1)  # proportional
        posterior_pdf = stats.beta.pdf(theta, alpha_post, beta_post)

        plt.figure(figsize=(10, 6))
        plt.plot(theta, prior_pdf, 'b--', label=f'Prior: Beta({alpha_prior},{beta_prior})', linewidth=2)
        plt.plot(theta, likelihood / np.max(likelihood) * np.max(posterior_pdf),
                'g:', label='Likelihood (scaled)', linewidth=2)
        plt.plot(theta, posterior_pdf, 'r-', label=f'Posterior: Beta({alpha_post},{beta_post})', linewidth=2)
        plt.axvline(post_mean, color='red', linestyle='--', alpha=0.5, label=f'Posterior mean={post_mean:.3f}')
        plt.axvline(mle, color='orange', linestyle='--', alpha=0.5, label=f'MLE={mle:.3f}')
        plt.xlabel('θ (probability of heads)')
        plt.ylabel('Density')
        plt.title('Beta-Binomial Conjugate Prior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/beta_binomial.png', dpi=100)
        print("\n[Plot saved to /tmp/beta_binomial.png]")
        plt.close()


def normal_normal_conjugate():
    """Demonstrate Normal-Normal conjugate prior."""
    print_section("3. Normal-Normal Conjugate Prior")

    print("Estimating mean μ of normal distribution (known variance)")

    # True data generation
    np.random.seed(42)
    true_mean = 100
    known_sigma = 15
    n = 20
    data = np.random.normal(true_mean, known_sigma, n)

    print(f"\nKnown: σ = {known_sigma}")
    print(f"Data: n = {n}, sample mean = {np.mean(data):.2f}")

    # Prior: N(μ₀, σ₀²)
    mu_0 = 110  # Prior belief
    sigma_0 = 20

    print(f"\nPrior: N({mu_0}, {sigma_0}²)")

    # Posterior: N(μₙ, σₙ²)
    # Precision formulation
    tau_0 = 1 / sigma_0**2  # Prior precision
    tau_likelihood = n / known_sigma**2  # Likelihood precision

    tau_post = tau_0 + tau_likelihood
    mu_post = (tau_0 * mu_0 + tau_likelihood * np.mean(data)) / tau_post
    sigma_post = np.sqrt(1 / tau_post)

    print(f"\nPosterior: N({mu_post:.2f}, {sigma_post:.2f}²)")
    print(f"  Posterior mean: {mu_post:.2f}")
    print(f"  Posterior std: {sigma_post:.2f}")

    # Credible interval
    ci = stats.norm.interval(0.95, mu_post, sigma_post)
    print(f"  95% Credible Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")

    # Compare with frequentist
    se = known_sigma / np.sqrt(n)
    freq_ci = stats.norm.interval(0.95, np.mean(data), se)

    print(f"\nFrequentist 95% CI: [{freq_ci[0]:.2f}, {freq_ci[1]:.2f}]")

    print(f"\nPrior influence:")
    print(f"  Prior mean: {mu_0}")
    print(f"  Sample mean: {np.mean(data):.2f}")
    print(f"  Posterior mean: {mu_post:.2f} (weighted average)")

    if HAS_PLT:
        x = np.linspace(70, 130, 300)
        prior_pdf = stats.norm.pdf(x, mu_0, sigma_0)
        likelihood_pdf = stats.norm.pdf(x, np.mean(data), known_sigma / np.sqrt(n))
        posterior_pdf = stats.norm.pdf(x, mu_post, sigma_post)

        plt.figure(figsize=(10, 6))
        plt.plot(x, prior_pdf, 'b--', label=f'Prior: N({mu_0},{sigma_0}²)', linewidth=2)
        plt.plot(x, likelihood_pdf, 'g:', label='Likelihood', linewidth=2)
        plt.plot(x, posterior_pdf, 'r-', label=f'Posterior: N({mu_post:.1f},{sigma_post:.1f}²)', linewidth=2)
        plt.axvline(mu_post, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('μ')
        plt.ylabel('Density')
        plt.title('Normal-Normal Conjugate Prior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/normal_normal.png', dpi=100)
        print("\n[Plot saved to /tmp/normal_normal.png]")
        plt.close()


def prior_influence():
    """Demonstrate how prior strength affects posterior."""
    print_section("4. Prior Influence on Posterior")

    print("Comparing weak vs strong priors")

    # Fixed data
    np.random.seed(123)
    n = 10
    data = np.random.binomial(1, 0.7, n)
    n_successes = np.sum(data)
    n_failures = n - n_successes

    print(f"\nData: {n_successes} successes in {n} trials")

    # Different priors
    priors = [
        ("Weak (uninformative)", 1, 1),
        ("Moderate", 5, 5),
        ("Strong (favor 0.5)", 20, 20)
    ]

    print(f"\nPosterior means with different priors:")

    for name, alpha_0, beta_0 in priors:
        alpha_post = alpha_0 + n_successes
        beta_post = beta_0 + n_failures
        post_mean = alpha_post / (alpha_post + beta_post)

        print(f"\n  {name}: Beta({alpha_0}, {beta_0})")
        print(f"    Posterior: Beta({alpha_post}, {beta_post})")
        print(f"    Posterior mean: {post_mean:.3f}")

    mle = n_successes / n
    print(f"\nMLE (no prior): {mle:.3f}")
    print(f"\nWith more data, all posteriors converge to MLE")


def credible_vs_confidence():
    """Compare Bayesian credible intervals with frequentist confidence intervals."""
    print_section("5. Credible vs Confidence Intervals")

    print("Interpretation differences:\n")

    print("Frequentist Confidence Interval:")
    print("  'If we repeat the experiment many times,")
    print("   95% of computed intervals will contain the true parameter'")
    print("  Parameter is FIXED, interval is RANDOM")

    print("\nBayesian Credible Interval:")
    print("  'The parameter has 95% probability of being in this interval")
    print("   given the observed data'")
    print("  Parameter is RANDOM, interval is FIXED (given data)")

    # Example with coin flips
    np.random.seed(456)
    n = 50
    true_theta = 0.6
    data = np.random.binomial(1, true_theta, n)
    n_heads = np.sum(data)

    # Frequentist CI
    p_hat = n_heads / n
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    freq_ci = stats.norm.interval(0.95, p_hat, se)

    # Bayesian CI (uniform prior)
    alpha_post = 1 + n_heads
    beta_post = 1 + (n - n_heads)
    bayes_ci = stats.beta.interval(0.95, alpha_post, beta_post)

    print(f"\n\nExample: {n_heads}/{n} heads observed")
    print(f"True θ = {true_theta}")

    print(f"\nFrequentist 95% CI: [{freq_ci[0]:.3f}, {freq_ci[1]:.3f}]")
    print(f"  Cannot say 'P(θ in interval) = 0.95'")

    print(f"\nBayesian 95% CI: [{bayes_ci[0]:.3f}, {bayes_ci[1]:.3f}]")
    print(f"  Can say 'P(θ in [{bayes_ci[0]:.3f}, {bayes_ci[1]:.3f}] | data) = 0.95'")

    print(f"\nBoth intervals contain true value: {freq_ci[0] <= true_theta <= freq_ci[1]}")


def sequential_updating():
    """Demonstrate sequential Bayesian updating."""
    print_section("6. Sequential Bayesian Updating")

    print("Updating beliefs as data arrives sequentially")

    # Start with prior
    alpha = 2
    beta = 2
    print(f"\nInitial prior: Beta({alpha}, {beta})")

    # Sequential data
    np.random.seed(789)
    true_p = 0.7
    batch_sizes = [5, 10, 20, 50]

    current_alpha = alpha
    current_beta = beta

    print(f"\nTrue probability: {true_p}")
    print(f"\nSequential updates:")

    for batch_size in batch_sizes:
        # New data
        data = np.random.binomial(1, true_p, batch_size)
        n_success = np.sum(data)
        n_fail = batch_size - n_success

        # Update
        current_alpha += n_success
        current_beta += n_fail

        post_mean = current_alpha / (current_alpha + current_beta)
        post_std = np.sqrt((current_alpha * current_beta) /
                          ((current_alpha + current_beta)**2 * (current_alpha + current_beta + 1)))

        print(f"\n  After {batch_size} more observations:")
        print(f"    New data: {n_success}/{batch_size} successes")
        print(f"    Posterior: Beta({current_alpha}, {current_beta})")
        print(f"    Mean: {post_mean:.4f}, Std: {post_std:.4f}")
        print(f"    Distance from truth: {abs(post_mean - true_p):.4f}")

    print(f"\n  Posterior converges to truth as data accumulates")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  BAYESIAN BASICS DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    bayes_theorem_discrete()
    beta_binomial_conjugate()
    normal_normal_conjugate()
    prior_influence()
    credible_vs_confidence()
    sequential_updating()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
