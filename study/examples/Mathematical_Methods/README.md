# Mathematical Methods Example Files

This directory contains 12 standalone Python scripts demonstrating mathematical methods commonly used in physics and engineering (based on Mary Boas' "Mathematical Methods in the Physical Sciences").

## Files Overview

### 01_infinite_series.py
- Series convergence tests (ratio, root, comparison)
- Partial sums and convergence visualization
- Taylor and Maclaurin series expansions
- Euler summation technique
- Power series radius of convergence

### 02_complex_numbers.py
- Complex arithmetic operations
- Polar and exponential forms
- Euler's formula
- Roots of unity
- Conformal mappings (z², 1/z, exp(z))
- Simple Mandelbrot set visualization

### 03_linear_algebra.py
- Matrix operations (addition, multiplication, transpose)
- Determinants and matrix inverse
- Eigenvalues and eigenvectors
- Matrix diagonalization
- Singular Value Decomposition (SVD)
- Solving linear systems
- Matrix exponential

### 05_vector_analysis.py
- Gradient of scalar fields
- Divergence of vector fields
- Curl of vector fields
- Line integrals
- Surface integrals
- Green's, Stokes', and Divergence theorem verification

### 06_fourier.py
- Fourier series coefficients
- Fast Fourier Transform (FFT)
- Spectral analysis
- Filtering in frequency domain
- Parseval's theorem
- Windowing techniques (Hanning, Hamming, Blackman)

### 07_ode.py
- Euler's method
- Runge-Kutta 4th order (RK4)
- scipy.integrate.solve_ivp
- Harmonic oscillator
- Damped oscillator
- Lorenz system (chaotic dynamics)
- Phase portraits

### 08_special_functions.py
- Bessel functions J_n(x) and Y_n(x)
- Legendre polynomials P_n(x)
- Hermite polynomials H_n(x)
- Laguerre polynomials L_n(x)
- Spherical harmonics Y_l^m(θ,φ)
- Gamma function Γ(x)
- Orthogonality properties

### 10_pde.py
- Heat equation (parabolic): finite difference method
- Wave equation (hyperbolic): explicit scheme
- Laplace equation (elliptic): Jacobi relaxation
- Poisson equation with source term
- Time evolution visualization

### 11_complex_analysis.py
- Cauchy integral formula (numerical)
- Residue theorem and computation
- Laurent series expansion
- Poles and essential singularities
- Analytic continuation concepts
- Conformal mapping visualization

### 12_laplace_transform.py
- Laplace transform pairs
- Inverse Laplace transform (Bromwich integral)
- Solving ODEs using Laplace transform
- Transfer functions
- Step response
- Frequency response (Bode plots)
- Convolution theorem

### 14_calculus_of_variations.py
- Euler-Lagrange equation
- Brachistochrone problem (fastest descent)
- Catenary curve (hanging chain)
- Geodesics on surfaces
- Lagrangian mechanics (pendulum, spring-mass)
- Minimal surface of revolution (catenoid)

### 15_tensors.py
- Tensor operations with numpy
- Index notation and Einstein summation (np.einsum)
- Metric tensor (Euclidean, polar, spherical)
- Christoffel symbols
- Coordinate transformations (Cartesian ↔ polar)
- Raising and lowering indices
- Levi-Civita tensor and cross product

## Requirements

All scripts are standalone and can be run independently. They require:

- **Required**: `numpy`
- **Optional**: `scipy` (for advanced numerical methods)
- **Optional**: `matplotlib` (for visualizations)

If scipy or matplotlib are not available, the scripts will run with reduced functionality and skip visualizations.

## Usage

Run any script directly:

```bash
python 01_infinite_series.py
python 02_complex_numbers.py
# ... etc
```

Each script:
- Prints detailed numerical results to console
- Generates visualizations (saved as PNG files) if matplotlib is available
- Includes docstrings explaining the mathematical concepts

## File Naming Convention

Files are numbered according to the chapter structure in Boas' textbook:
- Chapter 1: Infinite Series (01)
- Chapter 2: Complex Numbers (02)
- Chapter 3: Linear Algebra (03)
- Chapter 5: Vector Analysis (05)
- Chapter 6: Fourier Analysis (06)
- Chapter 7: ODEs (07)
- Chapter 8: Special Functions (08)
- Chapter 10: PDEs (10)
- Chapter 11: Complex Analysis (11)
- Chapter 12: Laplace Transform (12)
- Chapter 14: Calculus of Variations (14)
- Chapter 15: Tensors (15)

Note: Not all chapters have example files (e.g., Chapter 4 was incorporated into Chapter 3).

## License

These examples are part of the 03_Study project and are released under the MIT License.
