"""
07_glm.py

Demonstrates Generalized Linear Models (GLM):
- Logistic regression from scratch
- Poisson regression
- Link functions
- Deviance
- AIC/BIC model comparison
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

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


def logistic_regression_scratch():
    """Demonstrate logistic regression from scratch."""
    print_section("1. Logistic Regression (From Scratch)")

    # Generate binary classification data
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # True coefficients
    true_beta = np.array([-0.5, 1.5, -1.0])
    z = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2
    prob = 1 / (1 + np.exp(-z))
    y = (np.random.uniform(0, 1, n) < prob).astype(int)

    print(f"Generated data: n = {n}")
    print(f"True coefficients: {true_beta}")
    print(f"Class balance: {np.sum(y)}/{n} positive")

    # Design matrix
    X = np.column_stack([np.ones(n), x1, x2])

    # Logistic function
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Negative log-likelihood
    def neg_log_likelihood(beta):
        z = X @ beta
        p = sigmoid(z)
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Optimize using scipy
    beta_init = np.zeros(3)
    result = minimize(neg_log_likelihood, beta_init, method='BFGS')

    beta_hat = result.x

    print(f"\nEstimated coefficients:")
    print(f"  β₀ (intercept): {beta_hat[0]:.4f}")
    print(f"  β₁ (x1): {beta_hat[1]:.4f}")
    print(f"  β₂ (x2): {beta_hat[2]:.4f}")

    # Predictions
    z_hat = X @ beta_hat
    prob_hat = sigmoid(z_hat)
    y_pred = (prob_hat > 0.5).astype(int)

    # Accuracy
    accuracy = np.mean(y == y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Confusion matrix
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    print(f"\nConfusion matrix:")
    print(f"              Predicted")
    print(f"              0    1")
    print(f"  Actual  0  {tn:3d}  {fp:3d}")
    print(f"          1  {fn:3d}  {tp:3d}")

    # Log-likelihood at solution
    log_lik = -neg_log_likelihood(beta_hat)
    print(f"\nLog-likelihood: {log_lik:.2f}")

    # Null model (intercept only)
    p_null = np.mean(y)
    log_lik_null = np.sum(y * np.log(p_null + 1e-15) + (1 - y) * np.log(1 - p_null + 1e-15))

    # McFadden's R²
    r2_mcfadden = 1 - log_lik / log_lik_null
    print(f"McFadden's R²: {r2_mcfadden:.4f}")

    if HAS_PLT:
        # Decision boundary
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                               np.linspace(x2_min, x2_max, 100))
        X_grid = np.column_stack([np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()])
        Z = sigmoid(X_grid @ beta_hat).reshape(xx1.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx1, xx2, Z, levels=20, alpha=0.6, cmap='RdYlBu_r')
        plt.colorbar(label='P(y=1)')
        plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)
        plt.scatter(x1[y == 0], x2[y == 0], c='blue', marker='o', alpha=0.6, label='Class 0')
        plt.scatter(x1[y == 1], x2[y == 1], c='red', marker='s', alpha=0.6, label='Class 1')
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/logistic_regression.png', dpi=100)
        print("\n[Plot saved to /tmp/logistic_regression.png]")
        plt.close()


def poisson_regression():
    """Demonstrate Poisson regression."""
    print_section("2. Poisson Regression")

    # Generate count data
    np.random.seed(123)
    n = 150
    x = np.random.uniform(0, 5, n)

    # True model: log(λ) = β₀ + β₁*x
    true_beta = np.array([0.5, 0.4])
    log_lambda = true_beta[0] + true_beta[1] * x
    lambda_true = np.exp(log_lambda)
    y = np.random.poisson(lambda_true)

    print(f"Generated Poisson data: n = {n}")
    print(f"True coefficients: {true_beta}")
    print(f"Response range: [{y.min()}, {y.max()}]")

    # Design matrix
    X = np.column_stack([np.ones(n), x])

    # Negative log-likelihood for Poisson
    def neg_log_likelihood(beta):
        log_lambda = X @ beta
        lambda_pred = np.exp(log_lambda)
        # Poisson log-likelihood: y*log(λ) - λ - log(y!)
        # We can ignore log(y!) for optimization
        return -np.sum(y * log_lambda - lambda_pred)

    # Optimize
    beta_init = np.zeros(2)
    result = minimize(neg_log_likelihood, beta_init, method='BFGS')
    beta_hat = result.x

    print(f"\nEstimated coefficients:")
    print(f"  β₀ (intercept): {beta_hat[0]:.4f}")
    print(f"  β₁ (slope): {beta_hat[1]:.4f}")

    # Predictions
    log_lambda_hat = X @ beta_hat
    lambda_hat = np.exp(log_lambda_hat)

    print(f"\nPredicted λ range: [{lambda_hat.min():.2f}, {lambda_hat.max():.2f}]")

    # Deviance
    # Saturated model: λ_i = y_i
    # Null model: λ_i = mean(y)
    y_safe = np.where(y == 0, 1, y)  # Avoid log(0)

    deviance_residuals = 2 * (y * np.log(y_safe / lambda_hat) - (y - lambda_hat))
    deviance = np.sum(deviance_residuals)

    lambda_null = np.mean(y)
    deviance_null = 2 * np.sum(y * np.log(y_safe / lambda_null) - (y - lambda_null))

    print(f"\nDeviance: {deviance:.2f}")
    print(f"Null deviance: {deviance_null:.2f}")

    # Pseudo R²
    r2 = 1 - deviance / deviance_null
    print(f"Pseudo R²: {r2:.4f}")

    if HAS_PLT:
        x_plot = np.linspace(0, 5, 100)
        X_plot = np.column_stack([np.ones(len(x_plot)), x_plot])
        lambda_plot = np.exp(X_plot @ beta_hat)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.6, label='Observed counts')
        plt.plot(x_plot, lambda_plot, 'r-', linewidth=2, label='Fitted λ(x)')
        plt.xlabel('x')
        plt.ylabel('Count')
        plt.title('Poisson Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/poisson_regression.png', dpi=100)
        print("\n[Plot saved to /tmp/poisson_regression.png]")
        plt.close()


def link_functions_demo():
    """Demonstrate different link functions."""
    print_section("3. Link Functions")

    print("Common link functions in GLMs:\n")

    x = np.linspace(-5, 5, 200)

    # Logit (logistic regression)
    logit_inverse = 1 / (1 + np.exp(-x))

    # Probit (alternative for binary)
    probit_inverse = stats.norm.cdf(x)

    # Log (Poisson regression)
    log_inverse = np.exp(x)

    # Identity (linear regression)
    identity_inverse = x

    print("1. Logit link: g(μ) = log(μ/(1-μ))")
    print("   Inverse: μ = 1/(1+exp(-η))")
    print("   Used in: Logistic regression\n")

    print("2. Probit link: g(μ) = Φ⁻¹(μ)")
    print("   Inverse: μ = Φ(η)")
    print("   Used in: Probit regression\n")

    print("3. Log link: g(μ) = log(μ)")
    print("   Inverse: μ = exp(η)")
    print("   Used in: Poisson regression\n")

    print("4. Identity link: g(μ) = μ")
    print("   Inverse: μ = η")
    print("   Used in: Linear regression\n")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(x, logit_inverse, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('η (linear predictor)')
        axes[0, 0].set_ylabel('μ (mean)')
        axes[0, 0].set_title('Logit: μ = 1/(1+exp(-η))')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([-0.1, 1.1])

        axes[0, 1].plot(x, probit_inverse, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('η (linear predictor)')
        axes[0, 1].set_ylabel('μ (mean)')
        axes[0, 1].set_title('Probit: μ = Φ(η)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([-0.1, 1.1])

        axes[1, 0].plot(x, log_inverse, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('η (linear predictor)')
        axes[1, 0].set_ylabel('μ (mean)')
        axes[1, 0].set_title('Log: μ = exp(η)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 20])

        axes[1, 1].plot(x, identity_inverse, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('η (linear predictor)')
        axes[1, 1].set_ylabel('μ (mean)')
        axes[1, 1].set_title('Identity: μ = η')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/link_functions.png', dpi=100)
        print("[Plot saved to /tmp/link_functions.png]")
        plt.close()


def deviance_analysis():
    """Demonstrate deviance analysis."""
    print_section("4. Deviance Analysis")

    # Generate data
    np.random.seed(456)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    true_beta = np.array([0.5, 1.2, -0.8])
    log_lambda = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2
    y = np.random.poisson(np.exp(log_lambda))

    print("Comparing nested Poisson regression models")

    # Model 1: Intercept only
    X1 = np.ones((n, 1))

    # Model 2: Intercept + x1
    X2 = np.column_stack([np.ones(n), x1])

    # Model 3: Intercept + x1 + x2
    X3 = np.column_stack([np.ones(n), x1, x2])

    models = [
        ("Null (intercept only)", X1),
        ("x1", X2),
        ("x1 + x2", X3)
    ]

    results = []

    for name, X in models:
        # Fit model
        def neg_ll(beta):
            log_lambda = X @ beta
            return -np.sum(y * log_lambda - np.exp(log_lambda))

        beta_init = np.zeros(X.shape[1])
        result = minimize(neg_ll, beta_init, method='BFGS')
        beta_hat = result.x

        # Calculate deviance
        log_lambda_hat = X @ beta_hat
        lambda_hat = np.exp(log_lambda_hat)
        y_safe = np.where(y == 0, 1, y)
        deviance = 2 * np.sum(y * np.log(y_safe / lambda_hat) - (y - lambda_hat))

        # AIC and BIC
        k = X.shape[1]  # number of parameters
        log_lik = -neg_ll(beta_hat)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        results.append({
            'name': name,
            'k': k,
            'deviance': deviance,
            'aic': aic,
            'bic': bic,
            'log_lik': log_lik
        })

    print("\nModel comparison:")
    print(f"{'Model':<20} {'k':>3} {'Deviance':>12} {'AIC':>12} {'BIC':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<20} {r['k']:>3} {r['deviance']:>12.2f} {r['aic']:>12.2f} {r['bic']:>12.2f}")

    # Likelihood ratio test (Model 1 vs Model 3)
    lr_statistic = -2 * (results[0]['log_lik'] - results[2]['log_lik'])
    df = results[2]['k'] - results[0]['k']
    p_value = 1 - stats.chi2.cdf(lr_statistic, df)

    print(f"\nLikelihood ratio test (Null vs Full):")
    print(f"  LR statistic: {lr_statistic:.4f}")
    print(f"  df: {df}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Full model significantly better (p < 0.05)")

    # Best model by AIC
    best_aic_idx = np.argmin([r['aic'] for r in results])
    print(f"\nBest model by AIC: {results[best_aic_idx]['name']}")


def model_comparison_aic_bic():
    """Demonstrate AIC/BIC model comparison."""
    print_section("5. AIC/BIC Model Comparison")

    # Generate data with specific structure
    np.random.seed(789)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    x4 = np.random.normal(0, 1, n)  # Irrelevant

    # True model: only x1 and x2 matter
    z = -1 + 1.5*x1 + 0.8*x2
    p = 1 / (1 + np.exp(-z))
    y = (np.random.uniform(0, 1, n) < p).astype(int)

    print(f"True model: logit(p) = -1 + 1.5*x1 + 0.8*x2")
    print(f"x3, x4 are irrelevant")

    # Candidate models
    models = [
        ("x1", np.column_stack([np.ones(n), x1])),
        ("x1 + x2", np.column_stack([np.ones(n), x1, x2])),
        ("x1 + x2 + x3", np.column_stack([np.ones(n), x1, x2, x3])),
        ("x1 + x2 + x3 + x4", np.column_stack([np.ones(n), x1, x2, x3, x4]))
    ]

    results = []

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    for name, X in models:
        # Fit
        def neg_ll(beta):
            p = sigmoid(X @ beta)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        beta_init = np.zeros(X.shape[1])
        result = minimize(neg_ll, beta_init, method='BFGS')

        k = X.shape[1]
        log_lik = -neg_ll(result.x)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        results.append({'name': name, 'k': k, 'aic': aic, 'bic': bic})

    print(f"\n{'Model':<20} {'k':>3} {'AIC':>12} {'BIC':>12}")
    print("-" * 50)

    for r in results:
        print(f"{r['name']:<20} {r['k']:>3} {r['aic']:>12.2f} {r['bic']:>12.2f}")

    best_aic_idx = np.argmin([r['aic'] for r in results])
    best_bic_idx = np.argmin([r['bic'] for r in results])

    print(f"\nBest model by AIC: {results[best_aic_idx]['name']}")
    print(f"Best model by BIC: {results[best_bic_idx]['name']}")

    print(f"\nInterpretation:")
    print(f"  AIC prefers predictive accuracy")
    print(f"  BIC penalizes complexity more (prefers simpler models)")
    print(f"  Both should select x1+x2 (true model)")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  GENERALIZED LINEAR MODELS DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    logistic_regression_scratch()
    poisson_regression()
    link_functions_demo()
    deviance_analysis()
    model_comparison_aic_bic()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
