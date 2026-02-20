"""
Infinite Series - Convergence Tests and Taylor/Maclaurin Expansions

This script demonstrates:
- Series convergence tests (ratio, root, comparison)
- Partial sums and convergence visualization
- Taylor and Maclaurin series expansions
- Euler summation technique
- Power series radius of convergence
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def ratio_test(a_n_func, n_max=100):
    """
    Apply ratio test for convergence: lim |a_{n+1}/a_n|
    If limit < 1: converges, > 1: diverges, = 1: inconclusive
    """
    ratios = []
    for n in range(1, n_max):
        a_n = a_n_func(n)
        a_n1 = a_n_func(n + 1)
        if abs(a_n) > 1e-15:
            ratios.append(abs(a_n1 / a_n))

    if ratios:
        limit = ratios[-1]
        return limit, ratios
    return None, []


def root_test(a_n_func, n_max=100):
    """
    Apply root test for convergence: lim |a_n|^(1/n)
    If limit < 1: converges, > 1: diverges, = 1: inconclusive
    """
    roots = []
    for n in range(1, n_max):
        a_n = a_n_func(n)
        if a_n != 0:
            roots.append(abs(a_n) ** (1/n))

    if roots:
        limit = roots[-1]
        return limit, roots
    return None, []


def partial_sum(a_n_func, n_terms):
    """Compute partial sum S_n = sum_{k=1}^{n} a_k"""
    total = sum(a_n_func(k) for k in range(1, n_terms + 1))
    return total


def taylor_series_exp(x, n_terms=10):
    """Taylor series for e^x centered at 0"""
    result = 0
    for n in range(n_terms):
        result += x**n / np.math.factorial(n)
    return result


def taylor_series_sin(x, n_terms=10):
    """Taylor series for sin(x) centered at 0"""
    result = 0
    for n in range(n_terms):
        sign = (-1)**n
        result += sign * x**(2*n + 1) / np.math.factorial(2*n + 1)
    return result


def taylor_series_cos(x, n_terms=10):
    """Taylor series for cos(x) centered at 0"""
    result = 0
    for n in range(n_terms):
        sign = (-1)**n
        result += sign * x**(2*n) / np.math.factorial(2*n)
    return result


def euler_summation(a_n_func, n_terms=50):
    """
    Euler summation for accelerating slowly convergent series
    Uses Euler transform: S ≈ (1/2)[S_even + S_odd]
    """
    partial_sums = [partial_sum(a_n_func, n) for n in range(1, n_terms + 1)]

    # Apply Euler transform iteratively
    transformed = partial_sums[:]
    for iteration in range(5):
        new_transform = []
        for i in range(len(transformed) - 1):
            new_transform.append((transformed[i] + transformed[i+1]) / 2)
        if not new_transform:
            break
        transformed = new_transform

    return transformed[-1] if transformed else partial_sums[-1]


def radius_of_convergence_ratio(c_n_func, n_test=100):
    """
    Compute radius of convergence using ratio test
    R = lim |c_n / c_{n+1}|
    """
    ratios = []
    for n in range(1, n_test):
        c_n = c_n_func(n)
        c_n1 = c_n_func(n + 1)
        if abs(c_n1) > 1e-15:
            ratios.append(abs(c_n / c_n1))

    if ratios:
        return ratios[-1]
    return None


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("INFINITE SERIES - CONVERGENCE TESTS AND TAYLOR EXPANSIONS")
print("=" * 70)

# Test 1: Ratio test on geometric series
print("\n1. RATIO TEST - Geometric series a_n = (1/2)^n")
print("-" * 70)
geometric = lambda n: (1/2)**n
limit, ratios = ratio_test(geometric, 50)
print(f"Ratio test limit: {limit:.6f}")
print(f"Expected: 0.5 (converges since < 1)")
print(f"Partial sum (50 terms): {partial_sum(geometric, 50):.6f}")
print(f"Exact sum: {1 / (1 - 0.5):.6f}")

# Test 2: Ratio test on divergent series
print("\n2. RATIO TEST - Divergent series a_n = n!")
print("-" * 70)
factorial_series = lambda n: np.math.factorial(n)
# This grows too fast, test on smaller range
limit_approx = factorial_series(11) / factorial_series(10)
print(f"Ratio |a_11 / a_10| = {limit_approx:.1f}")
print(f"Expected: grows to infinity (diverges)")

# Test 3: Root test
print("\n3. ROOT TEST - Series a_n = (1/3)^n")
print("-" * 70)
series_root = lambda n: (1/3)**n
limit, roots = root_test(series_root, 50)
print(f"Root test limit: {limit:.6f}")
print(f"Expected: 0.333... (converges since < 1)")

# Test 4: Alternating harmonic series
print("\n4. ALTERNATING HARMONIC SERIES - sum (-1)^{n+1}/n")
print("-" * 70)
alt_harmonic = lambda n: ((-1)**(n+1)) / n
s_100 = partial_sum(alt_harmonic, 100)
s_1000 = partial_sum(alt_harmonic, 1000)
print(f"Partial sum (100 terms): {s_100:.6f}")
print(f"Partial sum (1000 terms): {s_1000:.6f}")
print(f"Converges to ln(2) = {np.log(2):.6f}")

# Test 5: Euler summation for ln(2)
print("\n5. EULER SUMMATION - Accelerating convergence to ln(2)")
print("-" * 70)
euler_result = euler_summation(alt_harmonic, 50)
print(f"Standard partial sum (50 terms): {partial_sum(alt_harmonic, 50):.6f}")
print(f"Euler summation (50 terms): {euler_result:.6f}")
print(f"Exact value ln(2): {np.log(2):.6f}")

# Test 6: Taylor series for e^x
print("\n6. TAYLOR SERIES - e^x at x=1")
print("-" * 70)
x_val = 1.0
for n in [5, 10, 20]:
    approx = taylor_series_exp(x_val, n)
    error = abs(approx - np.exp(x_val))
    print(f"n={n:2d} terms: {approx:.10f}, error: {error:.2e}")
print(f"Exact e^1: {np.exp(x_val):.10f}")

# Test 7: Taylor series for sin(x)
print("\n7. TAYLOR SERIES - sin(x) at x=π/4")
print("-" * 70)
x_val = np.pi / 4
for n in [3, 5, 10]:
    approx = taylor_series_sin(x_val, n)
    error = abs(approx - np.sin(x_val))
    print(f"n={n:2d} terms: {approx:.10f}, error: {error:.2e}")
print(f"Exact sin(π/4): {np.sin(x_val):.10f}")

# Test 8: Taylor series for cos(x)
print("\n8. TAYLOR SERIES - cos(x) at x=π/6")
print("-" * 70)
x_val = np.pi / 6
for n in [3, 5, 10]:
    approx = taylor_series_cos(x_val, n)
    error = abs(approx - np.cos(x_val))
    print(f"n={n:2d} terms: {approx:.10f}, error: {error:.2e}")
print(f"Exact cos(π/6): {np.cos(x_val):.10f}")

# Test 9: Radius of convergence
print("\n9. RADIUS OF CONVERGENCE - Power series sum c_n x^n")
print("-" * 70)
# Series: sum (1/n^2) x^n
c_n = lambda n: 1 / n**2
R = radius_of_convergence_ratio(c_n, 100)
print(f"Series: sum (1/n^2) x^n")
print(f"Radius of convergence R = {R:.6f}")
print(f"Expected: R = 1")

# Series: sum (n!) x^n
c_n_fact = lambda n: np.math.factorial(n) if n <= 20 else np.inf
print(f"\nSeries: sum (n!) x^n")
print(f"Radius of convergence R = 0 (series diverges for all x ≠ 0)")

# Series: sum (1/n!) x^n (exponential)
c_n_exp = lambda n: 1 / np.math.factorial(n)
R_exp = radius_of_convergence_ratio(c_n_exp, 50)
print(f"\nSeries: sum (1/n!) x^n")
print(f"Radius of convergence R → ∞ (ratio → 0)")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Convergence of geometric series
    ax = axes[0, 0]
    n_vals = np.arange(1, 51)
    partial_sums = [partial_sum(geometric, n) for n in n_vals]
    ax.plot(n_vals, partial_sums, 'b-', linewidth=2, label='Partial sums')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Limit = 1')
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('Partial sum')
    ax.set_title('Convergence: Geometric series (1/2)^n')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Taylor series for e^x
    ax = axes[0, 1]
    x_range = np.linspace(-2, 2, 100)
    ax.plot(x_range, np.exp(x_range), 'k-', linewidth=2, label='e^x (exact)')
    for n in [3, 5, 10]:
        y_approx = [taylor_series_exp(x, n) for x in x_range]
        ax.plot(x_range, y_approx, '--', label=f'n={n} terms')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Taylor Series: e^x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 8])

    # Plot 3: Taylor series for sin(x)
    ax = axes[1, 0]
    x_range = np.linspace(-2*np.pi, 2*np.pi, 200)
    ax.plot(x_range, np.sin(x_range), 'k-', linewidth=2, label='sin(x) exact')
    for n in [2, 4, 6]:
        y_approx = [taylor_series_sin(x, n) for x in x_range]
        ax.plot(x_range, y_approx, '--', label=f'n={n} terms')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Taylor Series: sin(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 2])

    # Plot 4: Convergence of alternating harmonic series
    ax = axes[1, 1]
    n_vals = np.arange(1, 101)
    partial_sums = [partial_sum(alt_harmonic, n) for n in n_vals]
    ax.plot(n_vals, partial_sums, 'b-', linewidth=1.5, label='Partial sums')
    ax.axhline(y=np.log(2), color='r', linestyle='--', label='Limit = ln(2)')
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('Partial sum')
    ax.set_title('Convergence: Alternating harmonic series')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/01_series_convergence.png', dpi=150)
    print("Saved visualization: 01_series_convergence.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
