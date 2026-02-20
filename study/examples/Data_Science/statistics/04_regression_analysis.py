"""
04_regression_analysis.py

Demonstrates regression analysis techniques:
- OLS regression from scratch
- Multiple regression
- Polynomial regression
- Residual analysis
- R-squared and adjusted R-squared
- Confidence and prediction intervals
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


def simple_ols_from_scratch():
    """Demonstrate OLS regression from scratch."""
    print_section("1. Simple OLS Regression (From Scratch)")

    # Generate data
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 10, n)
    true_slope = 2.5
    true_intercept = 5.0
    noise = np.random.normal(0, 2, n)
    y = true_intercept + true_slope * x + noise

    print(f"Generated data: n = {n}")
    print(f"True model: y = {true_intercept} + {true_slope}*x + ε")

    # Calculate OLS estimates from scratch
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Slope: β₁ = Cov(x,y) / Var(x)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    beta_1 = numerator / denominator

    # Intercept: β₀ = ȳ - β₁*x̄
    beta_0 = y_mean - beta_1 * x_mean

    print(f"\nOLS estimates:")
    print(f"  β₀ (intercept): {beta_0:.4f}")
    print(f"  β₁ (slope): {beta_1:.4f}")

    # Predictions and residuals
    y_pred = beta_0 + beta_1 * x
    residuals = y - y_pred

    # Sum of squares
    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum(residuals**2)
    ss_regression = np.sum((y_pred - y_mean)**2)

    print(f"\nSum of squares:")
    print(f"  SS_total: {ss_total:.2f}")
    print(f"  SS_regression: {ss_regression:.2f}")
    print(f"  SS_residual: {ss_residual:.2f}")

    # R-squared
    r_squared = 1 - (ss_residual / ss_total)
    print(f"\nR²: {r_squared:.4f}")

    # Standard error of regression
    dof = n - 2
    se_regression = np.sqrt(ss_residual / dof)
    print(f"Standard error of regression: {se_regression:.4f}")

    # Standard errors of coefficients
    se_beta_1 = se_regression / np.sqrt(denominator)
    se_beta_0 = se_regression * np.sqrt(1/n + x_mean**2/denominator)

    print(f"\nStandard errors:")
    print(f"  SE(β₀): {se_beta_0:.4f}")
    print(f"  SE(β₁): {se_beta_1:.4f}")

    # t-statistics and p-values
    t_beta_0 = beta_0 / se_beta_0
    t_beta_1 = beta_1 / se_beta_1
    p_beta_0 = 2 * (1 - stats.t.cdf(abs(t_beta_0), dof))
    p_beta_1 = 2 * (1 - stats.t.cdf(abs(t_beta_1), dof))

    print(f"\nt-statistics:")
    print(f"  t(β₀): {t_beta_0:.4f} (p={p_beta_0:.6f})")
    print(f"  t(β₁): {t_beta_1:.4f} (p={p_beta_1:.6f})")

    if HAS_PLT:
        plt.figure(figsize=(10, 5))
        plt.scatter(x, y, alpha=0.6, label='Data')
        plt.plot(x, y_pred, 'r-', linewidth=2, label=f'Fit: y={beta_0:.2f}+{beta_1:.2f}x')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/simple_regression.png', dpi=100)
        print("\n[Plot saved to /tmp/simple_regression.png]")
        plt.close()


def multiple_regression():
    """Demonstrate multiple regression."""
    print_section("2. Multiple Regression")

    # Generate data with multiple predictors
    np.random.seed(123)
    n = 150
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    x3 = np.random.uniform(-2, 2, n)

    # True model: y = 3 + 2*x1 - 1.5*x2 + 0.8*x3 + noise
    true_coeffs = np.array([3.0, 2.0, -1.5, 0.8])
    noise = np.random.normal(0, 1.5, n)
    y = true_coeffs[0] + true_coeffs[1]*x1 + true_coeffs[2]*x2 + true_coeffs[3]*x3 + noise

    print(f"Generated data: n = {n}")
    print(f"True coefficients: {true_coeffs}")

    # Design matrix
    X = np.column_stack([np.ones(n), x1, x2, x3])

    # OLS solution: β = (X'X)⁻¹X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    print(f"\nEstimated coefficients:")
    print(f"  β₀ (intercept): {beta[0]:.4f}")
    print(f"  β₁ (x1): {beta[1]:.4f}")
    print(f"  β₂ (x2): {beta[2]:.4f}")
    print(f"  β₃ (x3): {beta[3]:.4f}")

    # Predictions and residuals
    y_pred = X @ beta
    residuals = y - y_pred

    # R-squared
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Adjusted R-squared
    k = X.shape[1] - 1  # number of predictors (excluding intercept)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    print(f"\nModel fit:")
    print(f"  R²: {r_squared:.4f}")
    print(f"  Adjusted R²: {adj_r_squared:.4f}")

    # Standard errors
    dof = n - X.shape[1]
    mse = ss_residual / dof
    cov_matrix = mse * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov_matrix))

    print(f"\nStandard errors:")
    for i, s in enumerate(se):
        print(f"  SE(β₁): {s:.4f}")

    # F-statistic for overall model
    ms_regression = (ss_total - ss_residual) / k
    ms_residual = ss_residual / dof
    f_statistic = ms_regression / ms_residual
    p_value_f = 1 - stats.f.cdf(f_statistic, k, dof)

    print(f"\nOverall model test:")
    print(f"  F-statistic: {f_statistic:.4f}")
    print(f"  p-value: {p_value_f:.6f}")


def polynomial_regression():
    """Demonstrate polynomial regression."""
    print_section("3. Polynomial Regression")

    # Generate non-linear data
    np.random.seed(456)
    n = 80
    x = np.random.uniform(-3, 3, n)
    y_true = 2 - 0.5*x + 1.5*x**2 - 0.3*x**3
    y = y_true + np.random.normal(0, 1, n)

    print(f"Generated non-linear data: n = {n}")

    # Fit polynomials of different degrees
    degrees = [1, 2, 3, 4]

    print(f"\nFitting polynomials of degree 1-4:")

    results = []
    for degree in degrees:
        # Create polynomial features
        X = np.column_stack([x**i for i in range(degree + 1)])

        # Fit
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta

        # Metrics
        residuals = y - y_pred
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total)

        k = degree
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # AIC and BIC
        log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(ss_residual/n) - n/2
        aic = 2 * (degree + 1) - 2 * log_likelihood
        bic = (degree + 1) * np.log(n) - 2 * log_likelihood

        results.append({
            'degree': degree,
            'r2': r_squared,
            'adj_r2': adj_r_squared,
            'aic': aic,
            'bic': bic,
            'beta': beta
        })

        print(f"\n  Degree {degree}:")
        print(f"    R²: {r_squared:.4f}")
        print(f"    Adjusted R²: {adj_r_squared:.4f}")
        print(f"    AIC: {aic:.2f}")
        print(f"    BIC: {bic:.2f}")

    # Find best model by AIC
    best_aic_idx = np.argmin([r['aic'] for r in results])
    print(f"\nBest model by AIC: degree {results[best_aic_idx]['degree']}")

    if HAS_PLT:
        x_plot = np.linspace(-3, 3, 200)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, result in enumerate(results):
            degree = result['degree']
            beta = result['beta']

            X_plot = np.column_stack([x_plot**i for i in range(degree + 1)])
            y_plot = X_plot @ beta

            axes[idx].scatter(x, y, alpha=0.5, s=20)
            axes[idx].plot(x_plot, y_plot, 'r-', linewidth=2)
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            axes[idx].set_title(f'Degree {degree} (R²={result["r2"]:.3f})')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/polynomial_regression.png', dpi=100)
        print("\n[Plot saved to /tmp/polynomial_regression.png]")
        plt.close()


def residual_analysis():
    """Demonstrate residual analysis."""
    print_section("4. Residual Analysis")

    # Generate data with heteroscedasticity
    np.random.seed(789)
    n = 100
    x = np.random.uniform(1, 10, n)
    y = 5 + 2*x + np.random.normal(0, 0.5*x, n)  # Variance increases with x

    # Fit model
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    residuals = y - y_pred

    print(f"Fitted model: y = {beta[0]:.2f} + {beta[1]:.2f}*x")

    # Standardized residuals
    residual_std = np.std(residuals, ddof=2)
    standardized_residuals = residuals / residual_std

    print(f"\nResidual statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std: {residual_std:.4f}")
    print(f"  Min: {np.min(residuals):.4f}")
    print(f"  Max: {np.max(residuals):.4f}")

    # Check normality (Shapiro-Wilk test)
    stat, p_value = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk normality test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"  Residuals appear normally distributed")
    else:
        print(f"  Residuals may not be normally distributed")

    # Durbin-Watson statistic (autocorrelation)
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"\nDurbin-Watson statistic: {dw:.4f}")
    print(f"  (Values near 2 indicate no autocorrelation)")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Histogram of Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals vs order
        axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Observation order')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Order')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/residual_analysis.png', dpi=100)
        print("\n[Plot saved to /tmp/residual_analysis.png]")
        plt.close()


def confidence_prediction_intervals():
    """Demonstrate confidence and prediction intervals."""
    print_section("5. Confidence and Prediction Intervals")

    # Generate data
    np.random.seed(999)
    n = 50
    x = np.random.uniform(0, 10, n)
    y = 5 + 2*x + np.random.normal(0, 2, n)

    # Fit model
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    residuals = y - y_pred

    # Calculate intervals
    dof = n - 2
    mse = np.sum(residuals**2) / dof
    XtX_inv = np.linalg.inv(X.T @ X)

    # Points for prediction
    x_new = np.linspace(0, 10, 100)
    X_new = np.column_stack([np.ones(len(x_new)), x_new])
    y_new = X_new @ beta

    # Confidence interval for mean response
    conf_se = np.sqrt(mse * np.sum((X_new @ XtX_inv) * X_new, axis=1))
    t_crit = stats.t.ppf(0.975, dof)
    conf_lower = y_new - t_crit * conf_se
    conf_upper = y_new + t_crit * conf_se

    # Prediction interval for new observations
    pred_se = np.sqrt(mse * (1 + np.sum((X_new @ XtX_inv) * X_new, axis=1)))
    pred_lower = y_new - t_crit * pred_se
    pred_upper = y_new + t_crit * pred_se

    print(f"Fitted model: y = {beta[0]:.2f} + {beta[1]:.2f}*x")
    print(f"\nInterval widths at x=5:")

    # Find index closest to x=5
    idx = np.argmin(np.abs(x_new - 5))

    print(f"  Confidence interval width: {conf_upper[idx] - conf_lower[idx]:.4f}")
    print(f"  Prediction interval width: {pred_upper[idx] - pred_lower[idx]:.4f}")
    print(f"\nPrediction intervals are wider (account for individual variation)")

    if HAS_PLT:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.6, label='Data')
        plt.plot(x_new, y_new, 'r-', linewidth=2, label='Fit')
        plt.fill_between(x_new, conf_lower, conf_upper, alpha=0.2,
                        color='red', label='95% Confidence Interval')
        plt.fill_between(x_new, pred_lower, pred_upper, alpha=0.1,
                        color='blue', label='95% Prediction Interval')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Confidence vs Prediction Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/intervals.png', dpi=100)
        print("\n[Plot saved to /tmp/intervals.png]")
        plt.close()


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  REGRESSION ANALYSIS DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    simple_ols_from_scratch()
    multiple_regression()
    polynomial_regression()
    residual_analysis()
    confidence_prediction_intervals()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
