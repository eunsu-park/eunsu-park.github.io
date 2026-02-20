"""
수치 적분 (Numerical Integration)
Numerical Integration Methods

정적분 ∫[a,b] f(x)dx 를 수치적으로 계산하는 방법들입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# =============================================================================
# 1. 직사각형 법 (Rectangular Rule)
# =============================================================================
def rectangular_left(f: Callable, a: float, b: float, n: int) -> float:
    """
    왼쪽 직사각형 법
    각 구간의 왼쪽 끝점에서 함수값으로 직사각형 높이 결정
    """
    h = (b - a) / n
    result = 0
    for i in range(n):
        x = a + i * h
        result += f(x)
    return h * result


def rectangular_right(f: Callable, a: float, b: float, n: int) -> float:
    """오른쪽 직사각형 법"""
    h = (b - a) / n
    result = 0
    for i in range(1, n + 1):
        x = a + i * h
        result += f(x)
    return h * result


def rectangular_midpoint(f: Callable, a: float, b: float, n: int) -> float:
    """
    중점 직사각형 법 (Midpoint Rule)
    오차: O(h²)
    """
    h = (b - a) / n
    result = 0
    for i in range(n):
        x = a + (i + 0.5) * h
        result += f(x)
    return h * result


# =============================================================================
# 2. 사다리꼴 법 (Trapezoidal Rule)
# =============================================================================
def trapezoidal(f: Callable, a: float, b: float, n: int) -> float:
    """
    사다리꼴 법
    각 구간을 사다리꼴로 근사
    오차: O(h²)

    공식: h/2 * [f(x₀) + 2*Σf(xᵢ) + f(xₙ)]
    """
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        result += 2 * f(x)

    return h * result / 2


def trapezoidal_vectorized(f: Callable, a: float, b: float, n: int) -> float:
    """사다리꼴 법 (NumPy 벡터화)"""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return np.trapz(y, x)


# =============================================================================
# 3. 심프슨 법 (Simpson's Rule)
# =============================================================================
def simpsons_rule(f: Callable, a: float, b: float, n: int) -> float:
    """
    심프슨 1/3 법칙
    각 구간을 2차 다항식으로 근사
    오차: O(h⁴) - 매우 정확

    조건: n은 짝수여야 함
    공식: h/3 * [f(x₀) + 4*Σf(홀수) + 2*Σf(짝수) + f(xₙ)]
    """
    if n % 2 != 0:
        n += 1  # 짝수로 조정

    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return h * result / 3


def simpsons_38(f: Callable, a: float, b: float, n: int) -> float:
    """
    심프슨 3/8 법칙
    3차 다항식 근사
    조건: n은 3의 배수여야 함
    """
    if n % 3 != 0:
        n = (n // 3 + 1) * 3

    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 3 == 0:
            result += 2 * f(x)
        else:
            result += 3 * f(x)

    return 3 * h * result / 8


# =============================================================================
# 4. 롬베르그 적분 (Romberg Integration)
# =============================================================================
def romberg(f: Callable, a: float, b: float, max_order: int = 5) -> float:
    """
    롬베르그 적분
    사다리꼴 법에 Richardson 외삽을 반복 적용
    매우 높은 정확도 달성 가능
    """
    R = np.zeros((max_order, max_order))

    # 첫 번째 열: 사다리꼴 법
    h = b - a
    R[0, 0] = h * (f(a) + f(b)) / 2

    for i in range(1, max_order):
        h = h / 2
        # 새로운 점들의 합
        sum_new = sum(f(a + (2*k - 1) * h) for k in range(1, 2**(i-1) + 1))
        R[i, 0] = R[i-1, 0] / 2 + h * sum_new

        # Richardson 외삽
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    return R[max_order-1, max_order-1]


# =============================================================================
# 5. 가우스 구적법 (Gaussian Quadrature)
# =============================================================================
def gauss_legendre(f: Callable, a: float, b: float, n: int = 5) -> float:
    """
    가우스-르장드르 구적법
    n개의 점으로 (2n-1)차 다항식까지 정확히 적분
    매우 효율적

    [-1, 1]에서 [a, b]로 변환 적용
    """
    # n개 노드와 가중치 (사전 계산된 값)
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # 구간 변환: [-1, 1] → [a, b]
    # x = (b-a)/2 * t + (a+b)/2
    # dx = (b-a)/2 * dt

    transformed_nodes = (b - a) / 2 * nodes + (a + b) / 2
    result = sum(w * f(x) for x, w in zip(transformed_nodes, weights))

    return (b - a) / 2 * result


# =============================================================================
# 6. 적응적 적분 (Adaptive Quadrature)
# =============================================================================
def adaptive_simpsons(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 50
) -> Tuple[float, int]:
    """
    적응적 심프슨 적분
    오차가 큰 구간을 재귀적으로 세분화
    """
    call_count = [0]

    def _adaptive(a, b, fa, fb, fc, S, tol, depth):
        call_count[0] += 1
        c = (a + b) / 2
        d = (a + c) / 2
        e = (c + b) / 2

        fd = f(d)
        fe = f(e)

        S_left = (c - a) / 6 * (fa + 4*fd + fc)
        S_right = (b - c) / 6 * (fc + 4*fe + fb)
        S_new = S_left + S_right

        # 오차 추정
        error = (S_new - S) / 15

        if depth >= max_depth or abs(error) <= tol:
            return S_new + error  # Richardson 외삽
        else:
            left = _adaptive(a, c, fa, fc, fd, S_left, tol/2, depth+1)
            right = _adaptive(c, b, fc, fb, fe, S_right, tol/2, depth+1)
            return left + right

    fa, fb = f(a), f(b)
    fc = f((a + b) / 2)
    S = (b - a) / 6 * (fa + 4*fc + fb)

    result = _adaptive(a, b, fa, fb, fc, S, tol, 0)
    return result, call_count[0]


# =============================================================================
# 오차 분석 및 시각화
# =============================================================================
def analyze_convergence(f: Callable, a: float, b: float, exact: float):
    """수렴 속도 분석"""
    ns = [4, 8, 16, 32, 64, 128, 256, 512]
    methods = {
        'Midpoint': rectangular_midpoint,
        'Trapezoidal': trapezoidal,
        'Simpson': simpsons_rule,
    }

    print("\n수렴 분석:")
    print("-" * 70)
    print(f"{'n':>6} | {'Midpoint':>14} | {'Trapezoidal':>14} | {'Simpson':>14}")
    print("-" * 70)

    errors = {name: [] for name in methods}

    for n in ns:
        row = f"{n:>6} |"
        for name, method in methods.items():
            result = method(f, a, b, n)
            error = abs(result - exact)
            errors[name].append(error)
            row += f" {error:>14.2e} |"
        print(row)

    return ns, errors


def plot_methods_comparison(f, a, b, n=10):
    """적분 방법 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_dense = np.linspace(a, b, 500)
    y_dense = f(x_dense)

    methods = [
        ('직사각형 (중점)', rectangular_midpoint),
        ('사다리꼴', trapezoidal),
        ('심프슨', simpsons_rule),
    ]

    # 함수와 적분 영역
    for ax, (name, method) in zip(axes.flat[:3], methods):
        ax.plot(x_dense, y_dense, 'b-', linewidth=2, label='f(x)')
        ax.fill_between(x_dense, y_dense, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

        h = (b - a) / n
        x_pts = np.linspace(a, b, n + 1)

        if 'mid' in name:
            for i in range(n):
                xm = a + (i + 0.5) * h
                ax.bar(xm, f(xm), width=h, alpha=0.5, edgecolor='r', fill=False)
        elif 'trap' in name.lower() or '사다리꼴' in name:
            for i in range(n):
                x0, x1 = x_pts[i], x_pts[i+1]
                ax.fill([x0, x1, x1, x0], [0, 0, f(x1), f(x0)], alpha=0.5, edgecolor='r', fill=False)

        result = method(f, a, b, n)
        ax.set_title(f'{name}: {result:.6f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)

    # 수렴 그래프
    ax = axes[1, 1]
    ns, errors = [4, 8, 16, 32, 64, 128], {name: [] for name, _ in methods}
    exact = 2.0  # ∫[0,π] sin(x)dx = 2

    for n in ns:
        for name, method in methods:
            errors[name].append(abs(method(f, a, b, n) - exact))

    for name, errs in errors.items():
        ax.loglog(ns, errs, 'o-', label=name)

    ax.set_xlabel('n (구간 수)')
    ax.set_ylabel('오차')
    ax.set_title('수렴 속도')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/numerical_integration.png', dpi=150)
    plt.close()
    print("그래프 저장: numerical_integration.png")


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("수치 적분 (Numerical Integration) 예제")
    print("=" * 60)

    # 예제 1: ∫[0,π] sin(x)dx = 2
    print("\n[예제 1] ∫[0,π] sin(x)dx = 2")
    print("-" * 50)

    f = np.sin
    a, b = 0, np.pi
    exact = 2.0

    n = 100
    results = {
        '중점 직사각형': rectangular_midpoint(f, a, b, n),
        '사다리꼴': trapezoidal(f, a, b, n),
        '심프슨 1/3': simpsons_rule(f, a, b, n),
        '롬베르그': romberg(f, a, b, 6),
        '가우스-르장드르 (5점)': gauss_legendre(f, a, b, 5),
    }

    print(f"정확값: {exact}")
    print(f"{'방법':<20} {'결과':<15} {'오차':<15}")
    print("-" * 50)
    for name, result in results.items():
        error = abs(result - exact)
        print(f"{name:<20} {result:<15.10f} {error:<15.2e}")

    # 예제 2: ∫[0,1] e^(-x²)dx ≈ 0.7468...
    print("\n[예제 2] ∫[0,1] e^(-x²)dx (가우스 적분)")
    print("-" * 50)

    f2 = lambda x: np.exp(-x**2)
    exact2 = 0.746824132812427  # 참값 (erf 사용)

    for n in [10, 50, 100]:
        trap = trapezoidal(f2, 0, 1, n)
        simp = simpsons_rule(f2, 0, 1, n)
        print(f"n={n:3d}: 사다리꼴={trap:.10f}, 심프슨={simp:.10f}")

    gauss = gauss_legendre(f2, 0, 1, 10)
    print(f"가우스-르장드르 (10점): {gauss:.10f}")
    print(f"정확값: {exact2:.10f}")

    # 예제 3: 적응적 적분
    print("\n[예제 3] 적응적 적분 (급변하는 함수)")
    print("-" * 50)

    f3 = lambda x: 1 / (1 + 100 * (x - 0.5)**2)  # 좁은 피크
    exact3 = 0.3141277802329  # ≈ π/10

    trap = trapezoidal(f3, 0, 1, 100)
    result, calls = adaptive_simpsons(f3, 0, 1, tol=1e-8)

    print(f"사다리꼴 (n=100): {trap:.10f}, 오차: {abs(trap - exact3):.2e}")
    print(f"적응적 심프슨:    {result:.10f}, 오차: {abs(result - exact3):.2e}, 호출: {calls}")

    # 시각화
    try:
        plot_methods_comparison(np.sin, 0, np.pi, 10)
    except Exception as e:
        print(f"그래프 생성 실패: {e}")

    # 수렴 분석
    print("\n" + "=" * 60)
    print("수렴 속도 분석 (∫[0,π] sin(x)dx)")
    analyze_convergence(np.sin, 0, np.pi, 2.0)

    print("\n" + "=" * 60)
    print("수치 적분 방법 비교")
    print("=" * 60)
    print("""
    | 방법          | 오차 차수 | 특징                          |
    |--------------|----------|-------------------------------|
    | 직사각형 (좌/우)| O(h)     | 가장 단순, 부정확             |
    | 중점 직사각형  | O(h²)    | 직사각형 중 가장 정확          |
    | 사다리꼴      | O(h²)    | 단순하고 효율적                |
    | 심프슨 1/3    | O(h⁴)    | 매우 정확, 부드러운 함수에 적합 |
    | 롬베르그      | ~O(h^2k) | Richardson 외삽, 고정확도      |
    | 가우스 구적   | ~O(h^2n) | 최소 점으로 최대 정확도         |
    | 적응적 적분   | 가변     | 급변 구간 자동 세분화          |

    실무 권장:
    - scipy.integrate.quad: 적응적 가우스 구적 (가장 일반적)
    - scipy.integrate.romberg: 롬베르그 적분
    - scipy.integrate.simps: 심프슨 법칙 (등간격 데이터)
    - scipy.integrate.trapz: 사다리꼴 (등간격 데이터)
    """)


if __name__ == "__main__":
    main()
