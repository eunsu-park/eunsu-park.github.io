"""
Complex Analysis - Cauchy Integral, Residues, Laurent Series, Analytic Continuation

This script demonstrates:
- Cauchy integral formula (numerical)
- Residue theorem and computation
- Laurent series expansion
- Poles and essential singularities
- Analytic continuation concepts
- Conformal mapping properties
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def cauchy_integral_formula(f, z0, radius=1.0, n_points=1000):
    """
    Cauchy integral formula: f(z0) = (1/2πi) ∮ f(z)/(z-z0) dz
    Integrate around circle |z - z0| = radius
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    z = z0 + radius * np.exp(1j * theta)
    dz = 1j * radius * np.exp(1j * theta)

    integrand = f(z) / (z - z0) * dz
    integral = np.sum(integrand) * (2*np.pi / n_points)

    # Divide by 2πi
    result = integral / (2j * np.pi)
    return result


def cauchy_derivative_formula(f, z0, n=1, radius=0.5, n_points=1000):
    """
    Cauchy formula for derivatives:
    f^(n)(z0) = (n!/(2πi)) ∮ f(z)/(z-z0)^(n+1) dz
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    z = z0 + radius * np.exp(1j * theta)
    dz = 1j * radius * np.exp(1j * theta)

    integrand = f(z) / (z - z0)**(n+1) * dz
    integral = np.sum(integrand) * (2*np.pi / n_points)

    # Multiply by n!/(2πi)
    result = np.math.factorial(n) * integral / (2j * np.pi)
    return result


def residue_at_pole(f, z0, order=1, h=1e-4):
    """
    Compute residue at pole z0 of order n
    Res(f, z0) = lim_{z→z0} (1/(n-1)!) d^(n-1)/dz^(n-1) [(z-z0)^n f(z)]
    """
    if order == 1:
        # Simple pole: Res = lim_{z→z0} (z-z0)f(z)
        z_near = z0 + h
        residue = (z_near - z0) * f(z_near)
        return residue
    else:
        # Higher order pole: use numerical differentiation
        g = lambda z: (z - z0)**order * f(z)

        # Numerical (n-1)th derivative
        derivative = g(z0)
        for k in range(1, order):
            # Use central difference
            derivative = (g(z0 + h) - g(z0 - h)) / (2 * h)

        residue = derivative / np.math.factorial(order - 1)
        return residue


def contour_integral(f, path_func, t_range, n_points=1000):
    """
    Compute contour integral ∫_C f(z) dz
    path_func: parametric path z(t)
    """
    t = np.linspace(t_range[0], t_range[1], n_points)
    dt = (t_range[1] - t_range[0]) / (n_points - 1)

    z = path_func(t)
    # Compute dz/dt numerically
    dz_dt = np.gradient(z, dt)

    integrand = f(z) * dz_dt
    integral = np.trapz(integrand, t)

    return integral


def laurent_series_coefficients(f, z0, r_inner, r_outer, n_terms=10):
    """
    Compute Laurent series coefficients:
    f(z) = ∑_{n=-∞}^{∞} a_n (z-z0)^n
    a_n = (1/2πi) ∮ f(z)/(z-z0)^(n+1) dz
    """
    coeffs = {}

    # Positive powers (analytic part)
    radius = (r_inner + r_outer) / 2
    n_points = 1000
    theta = np.linspace(0, 2*np.pi, n_points)

    for n in range(-n_terms, n_terms + 1):
        z = z0 + radius * np.exp(1j * theta)
        dz = 1j * radius * np.exp(1j * theta)

        integrand = f(z) / (z - z0)**(n+1) * dz
        integral = np.sum(integrand) * (2*np.pi / n_points)

        coeffs[n] = integral / (2j * np.pi)

    return coeffs


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("COMPLEX ANALYSIS - CAUCHY, RESIDUES, LAURENT SERIES")
print("=" * 70)

# Test 1: Cauchy integral formula
print("\n1. CAUCHY INTEGRAL FORMULA")
print("-" * 70)

# f(z) = z^2
f_poly = lambda z: z**2
z0 = 1 + 1j

result = cauchy_integral_formula(f_poly, z0, radius=0.5)
exact = f_poly(z0)

print(f"f(z) = z²")
print(f"z₀ = {z0}")
print(f"Cauchy formula: f(z₀) = {result:.6f}")
print(f"Direct evaluation:     {exact:.6f}")
print(f"Error: {abs(result - exact):.2e}")

# f(z) = e^z
f_exp = lambda z: np.exp(z)
z0 = 0 + 0j

result = cauchy_integral_formula(f_exp, z0, radius=1.0)
exact = f_exp(z0)

print(f"\nf(z) = e^z")
print(f"z₀ = {z0}")
print(f"Cauchy formula: f(z₀) = {result:.6f}")
print(f"Direct evaluation:     {exact:.6f}")
print(f"Error: {abs(result - exact):.2e}")

# Test 2: Cauchy derivative formula
print("\n2. CAUCHY DERIVATIVE FORMULA")
print("-" * 70)

f_exp = lambda z: np.exp(z)
z0 = 0 + 0j

for n in range(1, 4):
    result = cauchy_derivative_formula(f_exp, z0, n=n, radius=0.5)
    exact = np.exp(z0)  # All derivatives of e^z equal e^z

    print(f"f^({n})(0) for f(z)=e^z:")
    print(f"  Cauchy formula: {result:.6f}")
    print(f"  Exact:          {exact:.6f}")
    print(f"  Error: {abs(result - exact):.2e}")

# Test 3: Residue theorem
print("\n3. RESIDUE THEOREM")
print("-" * 70)

# f(z) = 1/z (simple pole at z=0)
print("f(z) = 1/z, simple pole at z=0")
f_simple_pole = lambda z: 1 / z if abs(z) > 1e-10 else np.inf
residue = residue_at_pole(f_simple_pole, 0 + 0j, order=1, h=0.01)
print(f"Residue at z=0: {residue:.6f}")
print(f"Expected: 1.0")

# Verify with contour integral around circle
circle_path = lambda t: 1.0 * np.exp(1j * t)
integral = contour_integral(f_simple_pole, circle_path, (0, 2*np.pi))
print(f"∮ f(z)dz around |z|=1: {integral:.6f}")
print(f"2πi × Residue = {2j * np.pi * residue:.6f}")

# f(z) = 1/(z-1)(z-2) with poles at z=1 and z=2
print("\nf(z) = 1/[(z-1)(z-2)]")
f_two_poles = lambda z: 1 / ((z - 1) * (z - 2)) if abs((z-1)*(z-2)) > 1e-10 else np.inf

# Residues at each pole
res_1 = residue_at_pole(lambda z: 1/(z-2), 1 + 0j, order=1, h=0.01)
res_2 = residue_at_pole(lambda z: 1/(z-1), 2 + 0j, order=1, h=0.01)

print(f"Residue at z=1: {res_1:.6f} (expected: -1.0)")
print(f"Residue at z=2: {res_2:.6f} (expected:  1.0)")

# Contour enclosing both poles
circle_large = lambda t: 3.0 * np.exp(1j * t)
integral_large = contour_integral(f_two_poles, circle_large, (0, 2*np.pi))
print(f"∮ f(z)dz around |z|=3: {integral_large:.6f}")
print(f"2πi × (Res₁ + Res₂) = {2j * np.pi * (res_1 + res_2):.6f}")

# Test 4: Laurent series
print("\n4. LAURENT SERIES")
print("-" * 70)

# f(z) = 1/z(z-1) expanded around z=0
print("f(z) = 1/[z(z-1)], expanded around z=0")
print("Region: 0 < |z| < 1")

f_laurent = lambda z: 1 / (z * (z - 1)) if abs(z * (z-1)) > 1e-10 else np.inf

# For this function: 1/[z(z-1)] = -1/z - 1/(z-1) = -1/z + 1/(1-z)
# In region 0 < |z| < 1: = -1/z + ∑(z^n) = -1/z + 1 + z + z² + ...

coeffs = laurent_series_coefficients(f_laurent, 0 + 0j, 0.1, 0.9, n_terms=5)

print("\nLaurent coefficients a_n:")
for n in range(-5, 6):
    if n in coeffs:
        coeff = coeffs[n]
        if abs(coeff) > 1e-6:
            print(f"  a_{n:2d} = {coeff.real:8.4f} + {coeff.imag:8.4f}i")

# Test 5: Essential singularity
print("\n5. ESSENTIAL SINGULARITY")
print("-" * 70)

# f(z) = e^(1/z) has essential singularity at z=0
print("f(z) = e^(1/z), essential singularity at z=0")

f_essential = lambda z: np.exp(1/z) if abs(z) > 1e-10 else np.inf

# Laurent series: e^(1/z) = ∑_{n=0}^∞ (1/z)^n / n! = ∑_{n=-∞}^0 z^n / (-n)!
coeffs_essential = laurent_series_coefficients(f_essential, 0 + 0j, 0.2, 0.8, n_terms=5)

print("\nLaurent coefficients (principal part is infinite):")
for n in range(-5, 1):
    if n in coeffs_essential:
        coeff = coeffs_essential[n]
        expected = 1 / np.math.factorial(-n) if n <= 0 else 0
        print(f"  a_{n:2d} = {coeff.real:10.6f} (expected: {expected:.6f})")

# Residue (a_{-1})
residue_essential = coeffs_essential.get(-1, 0)
print(f"\nResidue (a_-1): {residue_essential.real:.6f}")
print(f"Expected: 1.0")

# Test 6: Conformal mapping properties
print("\n6. CONFORMAL MAPPING PROPERTIES")
print("-" * 70)

# Test that analytic functions preserve angles
# f(z) = z²
f_map = lambda z: z**2

z0 = 1 + 0.5j
h = 0.01

# Two directions from z0
direction1 = h * (1 + 0j)  # Horizontal
direction2 = h * (0 + 1j)  # Vertical

# Map to w-plane
w0 = f_map(z0)
w1 = f_map(z0 + direction1)
w2 = f_map(z0 + direction2)

# Angle in z-plane
angle_z = np.angle(direction2 / direction1)

# Angle in w-plane
dw1 = w1 - w0
dw2 = w2 - w0
angle_w = np.angle(dw2 / dw1)

print(f"Mapping: w = z²")
print(f"Point: z₀ = {z0}")
print(f"\nAngles between directions:")
print(f"  In z-plane: {np.degrees(angle_z):.2f}°")
print(f"  In w-plane: {np.degrees(angle_w):.2f}°")
print(f"  Difference: {np.degrees(abs(angle_z - angle_w)):.2f}° (should be ≈0)")

# Test 7: Analytic continuation
print("\n7. ANALYTIC CONTINUATION CONCEPT")
print("-" * 70)

# Example: f(z) = ∑ z^n (geometric series)
# Converges for |z| < 1, equals 1/(1-z)
# Can be continued to entire complex plane except z=1

print("f(z) = ∑_{n=0}^∞ z^n, |z| < 1")
print("Analytic continuation: f(z) = 1/(1-z), z ≠ 1")

z_inside = 0.5 + 0j
z_outside = 1.5 + 0j

# Series approximation (works only inside disk)
def geometric_series(z, n_terms=50):
    if abs(z) >= 1:
        return np.nan
    return sum(z**n for n in range(n_terms))

series_val = geometric_series(z_inside)
continued_val = 1 / (1 - z_inside)

print(f"\nInside disk |z| < 1:")
print(f"  z = {z_inside}")
print(f"  Series sum:    {series_val:.6f}")
print(f"  Continuation:  {continued_val:.6f}")

print(f"\nOutside disk |z| > 1:")
print(f"  z = {z_outside}")
print(f"  Series: diverges")
print(f"  Continuation: 1/(1-z) = {1/(1-z_outside):.6f}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Contour for Cauchy integral
    ax1 = plt.subplot(2, 3, 1)
    z0 = 1 + 0.5j
    radius = 0.7
    theta = np.linspace(0, 2*np.pi, 100)
    z_circle = z0 + radius * np.exp(1j * theta)

    ax1.plot(z_circle.real, z_circle.imag, 'b-', linewidth=2, label='Contour')
    ax1.plot(z0.real, z0.imag, 'ro', markersize=10, label='z₀')
    ax1.arrow(z_circle.real[25], z_circle.imag[25],
             z_circle.real[26]-z_circle.real[25],
             z_circle.imag[26]-z_circle.imag[25],
             head_width=0.1, head_length=0.05, fc='blue', ec='blue')
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    ax1.set_title('Cauchy Integral Formula Contour')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Poles and residues
    ax2 = plt.subplot(2, 3, 2)
    poles = [1 + 0j, 2 + 0j]
    circle_large = 3.0 * np.exp(1j * theta)

    ax2.plot(circle_large.real, circle_large.imag, 'b-', linewidth=2, label='Contour')
    for i, pole in enumerate(poles):
        ax2.plot(pole.real, pole.imag, 'rx', markersize=15, markeredgewidth=3)
        ax2.text(pole.real, pole.imag + 0.3, f'z={pole.real:.0f}', ha='center')

    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.set_title('Residue Theorem: f(z)=1/[(z-1)(z-2)]')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)

    # Plot 3: Laurent series annulus
    ax3 = plt.subplot(2, 3, 3)
    r_inner = 0.3
    r_outer = 0.9
    circle_inner = r_inner * np.exp(1j * theta)
    circle_outer = r_outer * np.exp(1j * theta)

    ax3.fill_between(circle_outer.real, circle_outer.imag,
                     alpha=0.3, label='Annulus')
    ax3.plot(circle_inner.real, circle_inner.imag, 'r--', linewidth=2)
    ax3.plot(circle_outer.real, circle_outer.imag, 'b--', linewidth=2)
    ax3.plot(0, 0, 'ko', markersize=8, label='Singularity')

    ax3.set_xlabel('Re(z)')
    ax3.set_ylabel('Im(z)')
    ax3.set_title('Laurent Series: Region of Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Plot 4: Conformal mapping w = z²
    ax4 = plt.subplot(2, 3, 4)

    # Grid in z-plane
    x_lines = np.linspace(-1.5, 1.5, 7)
    y_lines = np.linspace(-1.5, 1.5, 7)

    for x in x_lines:
        y = np.linspace(-1.5, 1.5, 100)
        z = x + 1j * y
        w = z**2
        ax4.plot(w.real, w.imag, 'b-', alpha=0.5, linewidth=0.8)

    for y in y_lines:
        x = np.linspace(-1.5, 1.5, 100)
        z = x + 1j * y
        w = z**2
        ax4.plot(w.real, w.imag, 'r-', alpha=0.5, linewidth=0.8)

    ax4.set_xlabel('Re(w)')
    ax4.set_ylabel('Im(w)')
    ax4.set_title('Conformal Map: w = z²')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # Plot 5: Analytic continuation
    ax5 = plt.subplot(2, 3, 5)

    # Unit circle (convergence of series)
    circle_unit = np.exp(1j * theta)
    ax5.plot(circle_unit.real, circle_unit.imag, 'b-', linewidth=2,
            label='|z| = 1 (series boundary)')
    ax5.fill(circle_unit.real, circle_unit.imag, alpha=0.2, color='blue')

    # Pole at z=1
    ax5.plot(1, 0, 'rx', markersize=15, markeredgewidth=3, label='Pole at z=1')

    # Sample points
    z_in = 0.5 + 0j
    z_out = 1.5 + 0j
    ax5.plot(z_in.real, z_in.imag, 'go', markersize=10, label='Series works')
    ax5.plot(z_out.real, z_out.imag, 'mo', markersize=10, label='Need continuation')

    ax5.set_xlabel('Re(z)')
    ax5.set_ylabel('Im(z)')
    ax5.set_title('Analytic Continuation: ∑z^n = 1/(1-z)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    ax5.set_xlim(-1.5, 2)
    ax5.set_ylim(-1.5, 1.5)

    # Plot 6: Branch cut (example: log(z))
    ax6 = plt.subplot(2, 3, 6)

    # Draw branch cut along negative real axis
    ax6.plot([-2, 0], [0, 0], 'r-', linewidth=4, label='Branch cut')
    ax6.plot(0, 0, 'ko', markersize=8, label='Branch point')

    # Contour avoiding branch cut
    theta_avoid = np.linspace(0.1, 2*np.pi - 0.1, 100)
    z_avoid = 1.5 * np.exp(1j * theta_avoid)
    ax6.plot(z_avoid.real, z_avoid.imag, 'b--', linewidth=2,
            label='Valid contour')

    ax6.set_xlabel('Re(z)')
    ax6.set_ylabel('Im(z)')
    ax6.set_title('Branch Cut: log(z)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    ax6.set_xlim(-2.5, 2.5)
    ax6.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/11_complex_analysis.png', dpi=150)
    print("Saved visualization: 11_complex_analysis.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
