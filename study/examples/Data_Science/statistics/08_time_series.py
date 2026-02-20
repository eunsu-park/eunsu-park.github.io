"""
08_time_series.py

Demonstrates time series analysis techniques:
- Moving average
- Exponential smoothing
- Autocorrelation
- Stationarity (ADF test concept)
- AR/MA/ARMA models
- Simple forecasting
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


def moving_average():
    """Demonstrate moving average smoothing."""
    print_section("1. Moving Average")

    # Generate time series with trend and noise
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 5, n)
    y = trend + seasonal + noise

    print(f"Generated time series: n = {n}")
    print(f"Components: trend + seasonal + noise")

    # Different window sizes
    windows = [5, 10, 20]

    print(f"\nMoving averages with different windows:")

    ma_results = {}
    for window in windows:
        # Simple moving average
        ma = np.convolve(y, np.ones(window) / window, mode='valid')
        ma_results[window] = ma

        # MSE compared to trend+seasonal (without noise)
        true_signal = (trend + seasonal)[window-1:]
        mse = np.mean((ma - true_signal)**2)

        print(f"\n  Window size {window}:")
        print(f"    Output length: {len(ma)}")
        print(f"    MSE (vs true signal): {mse:.2f}")
        print(f"    Mean: {np.mean(ma):.2f}")
        print(f"    Std: {np.std(ma, ddof=1):.2f}")

    if HAS_PLT:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, y, 'gray', alpha=0.5, label='Original')
        plt.plot(t, trend + seasonal, 'black', linewidth=2, label='True signal')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Original Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        for window in windows:
            ma = ma_results[window]
            t_ma = t[window-1:]
            plt.plot(t_ma, ma, label=f'MA({window})', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Moving Averages')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/moving_average.png', dpi=100)
        print("\n[Plot saved to /tmp/moving_average.png]")
        plt.close()


def exponential_smoothing():
    """Demonstrate exponential smoothing."""
    print_section("2. Exponential Smoothing")

    # Generate time series
    np.random.seed(123)
    n = 150
    t = np.arange(n)
    level = 50 + 0.3 * t
    noise = np.random.normal(0, 3, n)
    y = level + noise

    print(f"Generated time series: n = {n}")

    # Different smoothing parameters
    alphas = [0.1, 0.3, 0.7, 0.9]

    print(f"\nExponential smoothing with different α:")

    es_results = {}
    for alpha in alphas:
        # Simple exponential smoothing
        smoothed = np.zeros(n)
        smoothed[0] = y[0]

        for i in range(1, n):
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]

        es_results[alpha] = smoothed

        # MSE
        mse = np.mean((smoothed - level)**2)

        print(f"\n  α = {alpha}:")
        print(f"    MSE: {mse:.2f}")
        print(f"    Final value: {smoothed[-1]:.2f}")

        # Lag (measure of responsiveness)
        lag = np.mean(level - smoothed)
        print(f"    Average lag: {lag:.2f}")

    print(f"\nSmaller α: more smoothing, less responsive")
    print(f"Larger α: less smoothing, more responsive")

    if HAS_PLT:
        plt.figure(figsize=(12, 6))
        plt.plot(t, y, 'gray', alpha=0.4, label='Original', linewidth=1)
        plt.plot(t, level, 'black', linestyle='--', linewidth=2, label='True level')

        for alpha in alphas:
            plt.plot(t, es_results[alpha], label=f'α={alpha}', linewidth=2)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Exponential Smoothing')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/exponential_smoothing.png', dpi=100)
        print("\n[Plot saved to /tmp/exponential_smoothing.png]")
        plt.close()


def autocorrelation_analysis():
    """Demonstrate autocorrelation analysis."""
    print_section("3. Autocorrelation Analysis")

    # Generate AR(1) process
    np.random.seed(456)
    n = 300
    phi = 0.7
    y = np.zeros(n)
    y[0] = np.random.normal(0, 1)

    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.normal(0, 1)

    print(f"Generated AR(1) process: y_t = {phi}*y_{{t-1}} + ε_t")
    print(f"Sample size: {n}")

    # Calculate autocorrelation function (ACF)
    max_lag = 20
    acf = np.zeros(max_lag + 1)

    y_mean = np.mean(y)
    var = np.var(y, ddof=1)

    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            cov = np.mean((y[:-lag] - y_mean) * (y[lag:] - y_mean))
            acf[lag] = cov / var

    print(f"\nAutocorrelation function (ACF):")
    print(f"  Lag    ACF    Theoretical")
    for lag in range(min(11, max_lag + 1)):
        theoretical = phi**lag  # For AR(1)
        print(f"  {lag:3d}  {acf[lag]:6.3f}    {theoretical:6.3f}")

    # Ljung-Box test for autocorrelation
    # Test statistic Q = n(n+2) * sum((ρ_k)^2 / (n-k))
    h = 10  # Test up to lag h
    Q = n * (n + 2) * np.sum(acf[1:h+1]**2 / (n - np.arange(1, h + 1)))
    p_value = 1 - stats.chi2.cdf(Q, h)

    print(f"\nLjung-Box test (H₀: no autocorrelation up to lag {h}):")
    print(f"  Q statistic: {Q:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  Significant autocorrelation detected")

    if HAS_PLT:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Time series
        axes[0].plot(y, linewidth=1)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'AR(1) Process (φ={phi})')
        axes[0].grid(True, alpha=0.3)

        # ACF
        axes[1].bar(range(max_lag + 1), acf, alpha=0.7)
        axes[1].plot(range(max_lag + 1), [phi**lag for lag in range(max_lag + 1)],
                    'r--', linewidth=2, label='Theoretical')
        # Confidence bands (approximate)
        conf_level = 1.96 / np.sqrt(n)
        axes[1].axhline(conf_level, color='blue', linestyle='--', alpha=0.5)
        axes[1].axhline(-conf_level, color='blue', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('ACF')
        axes[1].set_title('Autocorrelation Function')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/autocorrelation.png', dpi=100)
        print("\n[Plot saved to /tmp/autocorrelation.png]")
        plt.close()


def stationarity_test():
    """Demonstrate stationarity testing concept."""
    print_section("4. Stationarity Testing")

    print("Comparing stationary vs non-stationary series\n")

    np.random.seed(789)
    n = 200

    # Stationary: AR(1) with |phi| < 1
    phi = 0.6
    y_stationary = np.zeros(n)
    for t in range(1, n):
        y_stationary[t] = phi * y_stationary[t-1] + np.random.normal(0, 1)

    # Non-stationary: Random walk
    y_rw = np.cumsum(np.random.normal(0, 1, n))

    print("Series 1: AR(1) with φ=0.6 (stationary)")
    print(f"  Mean: {np.mean(y_stationary):.2f}")
    print(f"  Variance: {np.var(y_stationary, ddof=1):.2f}")

    print("\nSeries 2: Random walk (non-stationary)")
    print(f"  Mean: {np.mean(y_rw):.2f}")
    print(f"  Variance: {np.var(y_rw, ddof=1):.2f}")

    # Simple stationarity check: variance in first vs second half
    n_half = n // 2

    var1_stat = np.var(y_stationary[:n_half], ddof=1)
    var2_stat = np.var(y_stationary[n_half:], ddof=1)

    var1_rw = np.var(y_rw[:n_half], ddof=1)
    var2_rw = np.var(y_rw[n_half:], ddof=1)

    print(f"\nVariance comparison (first half vs second half):")
    print(f"  Stationary series: {var1_stat:.2f} vs {var2_stat:.2f} (ratio: {var2_stat/var1_stat:.2f})")
    print(f"  Random walk: {var1_rw:.2f} vs {var2_rw:.2f} (ratio: {var2_rw/var1_rw:.2f})")

    print(f"\nStationary series: variance stable")
    print(f"Random walk: variance increases over time")

    # Differencing the random walk
    y_rw_diff = np.diff(y_rw)

    print(f"\nDifferenced random walk:")
    print(f"  Variance: {np.var(y_rw_diff, ddof=1):.2f}")
    print(f"  (Differencing makes random walk stationary)")


def ar_ma_models():
    """Demonstrate AR, MA, and ARMA models."""
    print_section("5. AR, MA, and ARMA Models")

    np.random.seed(111)
    n = 300

    # AR(1) model: y_t = 0.7*y_{t-1} + ε_t
    print("AR(1) model: y_t = 0.7*y_{t-1} + ε_t")
    phi = 0.7
    y_ar = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        y_ar[t] = phi * y_ar[t-1] + eps[t]

    print(f"  Mean: {np.mean(y_ar):.2f}")
    print(f"  Variance: {np.var(y_ar, ddof=1):.2f}")
    print(f"  Theoretical variance: {1/(1-phi**2):.2f}")

    # MA(1) model: y_t = ε_t + 0.6*ε_{t-1}
    print(f"\nMA(1) model: y_t = ε_t + 0.6*ε_{{t-1}}")
    theta = 0.6
    y_ma = np.zeros(n)
    for t in range(1, n):
        y_ma[t] = eps[t] + theta * eps[t-1]

    print(f"  Mean: {np.mean(y_ma):.2f}")
    print(f"  Variance: {np.var(y_ma, ddof=1):.2f}")
    print(f"  Theoretical variance: {1 + theta**2:.2f}")

    # ARMA(1,1) model: y_t = 0.5*y_{t-1} + ε_t + 0.4*ε_{t-1}
    print(f"\nARMA(1,1) model: y_t = 0.5*y_{{t-1}} + ε_t + 0.4*ε_{{t-1}}")
    phi_arma = 0.5
    theta_arma = 0.4
    y_arma = np.zeros(n)
    for t in range(1, n):
        y_arma[t] = phi_arma * y_arma[t-1] + eps[t] + theta_arma * eps[t-1]

    print(f"  Mean: {np.mean(y_arma):.2f}")
    print(f"  Variance: {np.var(y_arma, ddof=1):.2f}")

    if HAS_PLT:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(y_ar, linewidth=1)
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'AR(1): y_t = {phi}*y_{{t-1}} + ε_t')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(y_ma, linewidth=1, color='orange')
        axes[1].set_ylabel('Value')
        axes[1].set_title(f'MA(1): y_t = ε_t + {theta}*ε_{{t-1}}')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(y_arma, linewidth=1, color='green')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Value')
        axes[2].set_title(f'ARMA(1,1): y_t = {phi_arma}*y_{{t-1}} + ε_t + {theta_arma}*ε_{{t-1}}')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/ar_ma_models.png', dpi=100)
        print("\n[Plot saved to /tmp/ar_ma_models.png]")
        plt.close()


def simple_forecasting():
    """Demonstrate simple forecasting methods."""
    print_section("6. Simple Forecasting")

    # Generate data
    np.random.seed(222)
    n_train = 100
    n_test = 20
    n_total = n_train + n_test

    # AR(1) process
    phi = 0.8
    y = np.zeros(n_total)
    for t in range(1, n_total):
        y[t] = phi * y[t-1] + np.random.normal(0, 1)

    y_train = y[:n_train]
    y_test = y[n_train:]

    print(f"Training data: {n_train} observations")
    print(f"Test data: {n_test} observations")

    # Method 1: Mean forecast
    forecast_mean = np.full(n_test, np.mean(y_train))

    # Method 2: Last value (naive forecast)
    forecast_naive = np.full(n_test, y_train[-1])

    # Method 3: AR(1) forecast (assuming phi is estimated)
    phi_hat = np.corrcoef(y_train[:-1], y_train[1:])[0, 1]
    forecast_ar = np.zeros(n_test)
    forecast_ar[0] = phi_hat * y_train[-1]
    for h in range(1, n_test):
        forecast_ar[h] = phi_hat * forecast_ar[h-1]

    print(f"\nEstimated φ: {phi_hat:.4f} (true: {phi})")

    # Calculate forecast errors
    methods = [
        ("Mean", forecast_mean),
        ("Naive", forecast_naive),
        ("AR(1)", forecast_ar)
    ]

    print(f"\nForecast performance:")
    print(f"  {'Method':<10} {'MAE':>8} {'RMSE':>8}")
    print("-" * 30)

    for name, forecast in methods:
        mae = np.mean(np.abs(y_test - forecast))
        rmse = np.sqrt(np.mean((y_test - forecast)**2))
        print(f"  {name:<10} {mae:>8.3f} {rmse:>8.3f}")

    if HAS_PLT:
        plt.figure(figsize=(12, 6))

        t_train = np.arange(n_train)
        t_test = np.arange(n_train, n_total)

        plt.plot(t_train, y_train, 'b-', linewidth=2, label='Training data')
        plt.plot(t_test, y_test, 'black', linewidth=2, marker='o', label='Test data')
        plt.plot(t_test, forecast_mean, 'g--', linewidth=2, label='Mean forecast')
        plt.plot(t_test, forecast_naive, 'orange', linestyle='--', linewidth=2, label='Naive forecast')
        plt.plot(t_test, forecast_ar, 'r--', linewidth=2, label='AR(1) forecast')

        plt.axvline(n_train - 0.5, color='gray', linestyle=':', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Forecasting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/tmp/forecasting.png', dpi=100)
        print("\n[Plot saved to /tmp/forecasting.png]")
        plt.close()


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  TIME SERIES ANALYSIS DEMONSTRATIONS")
    print("=" * 70)

    np.random.seed(42)

    moving_average()
    exponential_smoothing()
    autocorrelation_analysis()
    stationarity_test()
    ar_ma_models()
    simple_forecasting()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
