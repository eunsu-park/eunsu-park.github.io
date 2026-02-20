"""
Complex Numbers - Arithmetic, Polar Form, and Conformal Mappings

This script demonstrates:
- Complex arithmetic operations
- Polar and exponential forms
- Euler's formula
- Roots of unity
- Conformal mappings (z^2, 1/z, exp(z))
- Simple Mandelbrot set visualization
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def complex_to_polar(z):
    """Convert complex number to polar form (r, theta)"""
    r = abs(z)
    theta = np.angle(z)
    return r, theta


def polar_to_complex(r, theta):
    """Convert polar form to complex number"""
    return r * np.exp(1j * theta)


def nth_roots_of_unity(n):
    """Compute n-th roots of unity: exp(2πik/n) for k=0,1,...,n-1"""
    roots = []
    for k in range(n):
        theta = 2 * np.pi * k / n
        root = np.exp(1j * theta)
        roots.append(root)
    return roots


def conformal_map_z_squared(z):
    """Conformal mapping w = z^2"""
    return z**2


def conformal_map_inverse(z):
    """Conformal mapping w = 1/z"""
    if z == 0:
        return np.inf
    return 1 / z


def conformal_map_exp(z):
    """Conformal mapping w = exp(z)"""
    return np.exp(z)


def mandelbrot_iteration(c, max_iter=100):
    """
    Mandelbrot iteration: z_{n+1} = z_n^2 + c
    Returns number of iterations before |z| > 2
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return max_iter


# ============================================================================
# MAIN DEMONSTRATIONS
# ============================================================================

print("=" * 70)
print("COMPLEX NUMBERS - ARITHMETIC, POLAR FORM, AND MAPPINGS")
print("=" * 70)

# Test 1: Complex arithmetic
print("\n1. COMPLEX ARITHMETIC")
print("-" * 70)
z1 = 3 + 4j
z2 = 1 - 2j
print(f"z1 = {z1}")
print(f"z2 = {z2}")
print(f"z1 + z2 = {z1 + z2}")
print(f"z1 - z2 = {z1 - z2}")
print(f"z1 * z2 = {z1 * z2}")
print(f"z1 / z2 = {z1 / z2:.4f}")
print(f"z1 conjugate = {np.conj(z1)}")
print(f"|z1| = {abs(z1):.4f}")

# Test 2: Polar form
print("\n2. POLAR FORM AND EULER'S FORMULA")
print("-" * 70)
z = 1 + 1j
r, theta = complex_to_polar(z)
print(f"z = {z}")
print(f"Polar form: r = {r:.4f}, θ = {theta:.4f} rad ({np.degrees(theta):.2f}°)")
print(f"Verification: r*e^(iθ) = {polar_to_complex(r, theta):.4f}")
print(f"\nEuler's formula: e^(iπ) + 1 = {np.exp(1j * np.pi) + 1:.10f}")
print(f"e^(iπ/2) = {np.exp(1j * np.pi / 2):.4f} (should be i)")

# Test 3: Roots of unity
print("\n3. ROOTS OF UNITY")
print("-" * 70)
for n in [3, 4, 5]:
    print(f"\n{n}-th roots of unity:")
    roots = nth_roots_of_unity(n)
    for k, root in enumerate(roots):
        r, theta = complex_to_polar(root)
        print(f"  ω_{k} = {root:.4f}, |ω| = {r:.4f}, arg = {np.degrees(theta):6.2f}°")
    # Verify they are roots
    verification = sum(root**n for root in roots)
    print(f"  Verification: sum(ω^{n}) = {verification:.6f} (should be {n})")

# Test 4: De Moivre's theorem
print("\n4. DE MOIVRE'S THEOREM")
print("-" * 70)
z = np.exp(1j * np.pi / 6)  # e^(iπ/6)
n = 3
z_n_direct = z**n
r, theta = complex_to_polar(z)
z_n_demoivre = polar_to_complex(r**n, n * theta)
print(f"z = e^(iπ/6) = {z:.4f}")
print(f"n = {n}")
print(f"z^n (direct) = {z_n_direct:.4f}")
print(f"z^n (De Moivre) = {z_n_demoivre:.4f}")
print(f"Expected: e^(iπ/2) = {np.exp(1j * np.pi / 2):.4f}")

# Test 5: Complex logarithm
print("\n5. COMPLEX LOGARITHM")
print("-" * 70)
z = 1 + 1j
log_z = np.log(z)
print(f"z = {z}")
print(f"ln(z) = {log_z:.4f}")
r, theta = complex_to_polar(z)
print(f"ln(z) = ln(r) + iθ = {np.log(r):.4f} + {theta:.4f}i")
print(f"Verification: e^(ln(z)) = {np.exp(log_z):.4f}")

# Test 6: Conformal mappings
print("\n6. CONFORMAL MAPPINGS")
print("-" * 70)

# w = z^2
z_test = 1 + 1j
w = conformal_map_z_squared(z_test)
print(f"\nMapping w = z^2:")
print(f"z = {z_test} → w = {w:.4f}")
r, theta = complex_to_polar(z_test)
r_w, theta_w = complex_to_polar(w)
print(f"|z| = {r:.4f}, arg(z) = {np.degrees(theta):.2f}°")
print(f"|w| = {r_w:.4f}, arg(w) = {np.degrees(theta_w):.2f}°")
print(f"Note: |w| = |z|^2, arg(w) = 2*arg(z)")

# w = 1/z
z_test = 2 + 0j
w = conformal_map_inverse(z_test)
print(f"\nMapping w = 1/z:")
print(f"z = {z_test} → w = {w:.4f}")
z_test = 1j
w = conformal_map_inverse(z_test)
print(f"z = {z_test} → w = {w:.4f}")

# w = exp(z)
z_test = 1 + 1j * np.pi
w = conformal_map_exp(z_test)
print(f"\nMapping w = exp(z):")
print(f"z = 1 + iπ → w = {w:.4f}")
print(f"Expected: e^1 * e^(iπ) = e * (-1) = {np.e * (-1):.4f}")

# Test 7: Mandelbrot set sample points
print("\n7. MANDELBROT SET - SAMPLE POINTS")
print("-" * 70)
test_points = [
    (0 + 0j, "Origin"),
    (-1 + 0j, "(-1, 0)"),
    (0.25 + 0j, "(0.25, 0)"),
    (-0.5 + 0.5j, "(-0.5, 0.5)"),
    (1 + 0j, "(1, 0) - outside"),
]

for c, description in test_points:
    iterations = mandelbrot_iteration(c, max_iter=100)
    in_set = "IN SET" if iterations == 100 else "ESCAPES"
    print(f"c = {c:>10} ({description:20s}): {iterations:3d} iterations - {in_set}")

# Visualization
if HAS_MATPLOTLIB:
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Roots of unity
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    for n, color in [(3, 'r'), (4, 'b'), (5, 'g')]:
        roots = nth_roots_of_unity(n)
        theta_vals = [np.angle(r) for r in roots]
        r_vals = [abs(r) for r in roots]
        ax1.plot(theta_vals + [theta_vals[0]], r_vals + [r_vals[0]],
                'o-', label=f'n={n}', markersize=8, color=color)
    ax1.set_title('Roots of Unity')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Conformal mapping z^2
    ax2 = plt.subplot(2, 3, 2)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)

    # Draw grid in z-plane
    for xi in x[::4]:
        z_line = xi + 1j * y
        w_line = conformal_map_z_squared(z_line)
        ax2.plot(w_line.real, w_line.imag, 'b-', alpha=0.5, linewidth=0.5)
    for yi in y[::4]:
        z_line = x + 1j * yi
        w_line = conformal_map_z_squared(z_line)
        ax2.plot(w_line.real, w_line.imag, 'r-', alpha=0.5, linewidth=0.5)

    ax2.set_xlabel('Re(w)')
    ax2.set_ylabel('Im(w)')
    ax2.set_title('Conformal Map: w = z²')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)

    # Plot 3: Conformal mapping 1/z
    ax3 = plt.subplot(2, 3, 3)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)

    for xi in x[::4]:
        if abs(xi) > 0.1:
            z_line = xi + 1j * y
            w_line = 1 / z_line
            w_line = w_line[np.abs(w_line) < 10]
            ax3.plot(w_line.real, w_line.imag, 'b-', alpha=0.5, linewidth=0.5)

    for yi in y[::4]:
        if abs(yi) > 0.1:
            z_line = x + 1j * yi
            w_line = 1 / z_line
            w_line = w_line[np.abs(w_line) < 10]
            ax3.plot(w_line.real, w_line.imag, 'r-', alpha=0.5, linewidth=0.5)

    ax3.set_xlabel('Re(w)')
    ax3.set_ylabel('Im(w)')
    ax3.set_title('Conformal Map: w = 1/z')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)

    # Plot 4: Mandelbrot set
    ax4 = plt.subplot(2, 3, 4)
    xmin, xmax, ymin, ymax = -2.5, 1.0, -1.25, 1.25
    width, height = 400, 300

    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.zeros((height, width))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            c = x + 1j * y
            mandelbrot_set[i, j] = mandelbrot_iteration(c, max_iter=50)

    ax4.imshow(mandelbrot_set, extent=[xmin, xmax, ymin, ymax],
               cmap='hot', origin='lower', interpolation='bilinear')
    ax4.set_xlabel('Re(c)')
    ax4.set_ylabel('Im(c)')
    ax4.set_title('Mandelbrot Set')

    # Plot 5: Complex exponential spiral
    ax5 = plt.subplot(2, 3, 5)
    t = np.linspace(0, 4*np.pi, 500)
    z = (0.1 + 0.1j) * t * np.exp(1j * t)
    ax5.plot(z.real, z.imag, 'b-', linewidth=1.5)
    ax5.set_xlabel('Re(z)')
    ax5.set_ylabel('Im(z)')
    ax5.set_title('Complex Exponential Spiral')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')

    # Plot 6: Euler's identity visualization
    ax6 = plt.subplot(2, 3, 6)
    theta = np.linspace(0, 2*np.pi, 100)
    z_circle = np.exp(1j * theta)
    ax6.plot(z_circle.real, z_circle.imag, 'b-', linewidth=2, label='|z|=1')

    # Mark special points
    special_points = [
        (0, 1, '0'),
        (np.pi/2, 1j, 'π/2'),
        (np.pi, -1, 'π'),
        (3*np.pi/2, -1j, '3π/2'),
    ]
    for angle, z, label in special_points:
        ax6.plot(z.real, z.imag, 'ro', markersize=8)
        ax6.text(z.real*1.2, z.imag*1.2, f'e^(i{label})', ha='center')

    ax6.set_xlabel('Real')
    ax6.set_ylabel('Imaginary')
    ax6.set_title("Euler's Formula: e^(iθ)")
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    ax6.legend()

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Mathematical_Methods/02_complex_numbers.png', dpi=150)
    print("Saved visualization: 02_complex_numbers.png")
    plt.close()

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
