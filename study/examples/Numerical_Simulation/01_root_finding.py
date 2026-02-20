"""
근 찾기 (Root Finding)
Numerical Root Finding Methods

f(x) = 0 을 만족하는 x를 수치적으로 찾는 방법들입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional


# =============================================================================
# 1. 이분법 (Bisection Method)
# =============================================================================
def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    이분법으로 f(x) = 0의 근 찾기

    조건: f(a)와 f(b)의 부호가 달라야 함 (중간값 정리)
    수렴 속도: 선형 (매 반복마다 구간 절반)

    Args:
        f: 목표 함수
        a, b: 초기 구간
        tol: 허용 오차
        max_iter: 최대 반복 횟수

    Returns:
        (근, 반복 횟수, 중간값 히스토리)
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a)와 f(b)의 부호가 달라야 합니다")

    history = []

    for i in range(max_iter):
        c = (a + b) / 2
        history.append(c)

        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i + 1, history

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, max_iter, history


# =============================================================================
# 2. 뉴턴-랩슨 방법 (Newton-Raphson Method)
# =============================================================================
def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    뉴턴-랩슨 방법으로 f(x) = 0의 근 찾기

    x_{n+1} = x_n - f(x_n) / f'(x_n)

    수렴 속도: 2차 (제곱 수렴)
    단점: 도함수 필요, 초기값에 민감

    Args:
        f: 목표 함수
        df: 도함수
        x0: 초기값
        tol: 허용 오차
        max_iter: 최대 반복 횟수

    Returns:
        (근, 반복 횟수, 히스토리)
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-15:
            raise ValueError("도함수가 0에 가까움: 발산 위험")

        x_new = x - fx / dfx
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, i + 1, history

        x = x_new

    return x, max_iter, history


# =============================================================================
# 3. 할선법 (Secant Method)
# =============================================================================
def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    할선법으로 f(x) = 0의 근 찾기

    뉴턴법의 도함수를 차분으로 근사:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    수렴 속도: 약 1.618차 (황금비)
    장점: 도함수 불필요

    Args:
        f: 목표 함수
        x0, x1: 두 초기값
        tol: 허용 오차
        max_iter: 최대 반복 횟수

    Returns:
        (근, 반복 횟수, 히스토리)
    """
    history = [x0, x1]

    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < 1e-15:
            raise ValueError("분모가 0에 가까움")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)

        if abs(x2 - x1) < tol:
            return x2, i + 1, history

        x0, x1 = x1, x2

    return x1, max_iter, history


# =============================================================================
# 4. 고정점 반복법 (Fixed-Point Iteration)
# =============================================================================
def fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int, list]:
    """
    고정점 반복: x = g(x)를 만족하는 x 찾기

    f(x) = 0 을 x = g(x) 형태로 변환
    예: x² - 2 = 0  →  x = 2/x 또는 x = (x + 2/x)/2

    수렴 조건: |g'(x*)| < 1

    Args:
        g: 반복 함수
        x0: 초기값
        tol: 허용 오차
        max_iter: 최대 반복 횟수

    Returns:
        (고정점, 반복 횟수, 히스토리)
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        x_new = g(x)
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, i + 1, history

        x = x_new

    return x, max_iter, history


# =============================================================================
# 5. Brent's Method (scipy와 비교)
# =============================================================================
def brents_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Tuple[float, int]:
    """
    Brent's Method (간소화 버전)
    이분법, 할선법, 역2차보간을 조합한 방법

    실무에서는 scipy.optimize.brentq 사용 권장
    """
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("f(a)와 f(b)의 부호가 달라야 합니다")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = c  # d 초기화 (첫 iteration에서 사용됨)
    mflag = True

    for i in range(max_iter):
        if fa != fc and fb != fc:
            # 역2차 보간
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # 할선법
            s = b - fb * (b - a) / (fb - fa)

        # 조건 체크 후 이분법으로 대체
        conditions = [
            not ((3 * a + b) / 4 <= s <= b or b <= s <= (3 * a + b) / 4),
            mflag and abs(s - b) >= abs(b - c) / 2,
            not mflag and abs(s - b) >= abs(c - d) / 2,
        ]

        if any(conditions):
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c, fc = c, b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(b - a) < tol or abs(fb) < tol:
            return b, i + 1

    return b, max_iter


# =============================================================================
# 시각화
# =============================================================================
def plot_convergence(f, methods_data, x_range, title="근 찾기 수렴 비교"):
    """수렴 과정 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 함수와 근
    x = np.linspace(x_range[0], x_range[1], 500)
    y = [f(xi) for xi in x]
    axes[0].plot(x, y, 'b-', label='f(x)')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    colors = plt.cm.tab10.colors
    for i, (name, root, _, history) in enumerate(methods_data):
        axes[0].scatter([root], [0], s=100, color=colors[i], zorder=5, label=f'{name}: x={root:.6f}')

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('함수와 근')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 오른쪽: 수렴 속도
    for i, (name, root, iters, history) in enumerate(methods_data):
        if history:
            errors = [abs(h - root) for h in history]
            errors = [e if e > 1e-16 else 1e-16 for e in errors]
            axes[1].semilogy(errors, 'o-', color=colors[i], label=f'{name} ({iters}회)')

    axes[1].set_xlabel('반복 횟수')
    axes[1].set_ylabel('오차 (log scale)')
    axes[1].set_title('수렴 속도 비교')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/root_finding.png', dpi=150)
    plt.close()
    print("    그래프 저장: root_finding.png")


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("근 찾기 (Root Finding) 예제")
    print("=" * 60)

    # 예제 1: f(x) = x³ - x - 2 = 0 (근 ≈ 1.5214)
    print("\n[예제 1] f(x) = x³ - x - 2 = 0")
    print("-" * 40)

    f = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1

    methods_data = []

    # 이분법
    root, iters, hist = bisection(f, 1, 2)
    methods_data.append(("Bisection", root, iters, hist))
    print(f"이분법:     근 = {root:.10f}, 반복 = {iters}")

    # 뉴턴-랩슨
    root, iters, hist = newton_raphson(f, df, 1.5)
    methods_data.append(("Newton", root, iters, hist))
    print(f"뉴턴-랩슨: 근 = {root:.10f}, 반복 = {iters}")

    # 할선법
    root, iters, hist = secant(f, 1, 2)
    methods_data.append(("Secant", root, iters, hist))
    print(f"할선법:     근 = {root:.10f}, 반복 = {iters}")

    # 시각화
    try:
        plot_convergence(f, methods_data, (0, 3), "f(x) = x³ - x - 2")
    except Exception as e:
        print(f"    그래프 생성 실패: {e}")

    # 예제 2: √2 구하기 (x² - 2 = 0)
    print("\n[예제 2] √2 구하기 (x² - 2 = 0)")
    print("-" * 40)

    f2 = lambda x: x**2 - 2
    df2 = lambda x: 2*x
    g = lambda x: (x + 2/x) / 2  # 바빌로니아 방법

    root, iters, _ = newton_raphson(f2, df2, 1.0)
    print(f"뉴턴-랩슨:   √2 = {root:.15f}, 반복 = {iters}")

    root, iters, _ = fixed_point(g, 1.0)
    print(f"고정점 반복: √2 = {root:.15f}, 반복 = {iters}")

    print(f"실제 √2:        {np.sqrt(2):.15f}")

    # 예제 3: cos(x) = x 고정점
    print("\n[예제 3] cos(x) = x (Dottie Number)")
    print("-" * 40)

    g_cos = lambda x: np.cos(x)
    root, iters, _ = fixed_point(g_cos, 0.5)
    print(f"고정점 x = cos(x): {root:.10f}, 반복 = {iters}")

    print("\n" + "=" * 60)
    print("근 찾기 방법 비교")
    print("=" * 60)
    print("""
    | 방법        | 수렴 속도 | 장점                | 단점                |
    |------------|----------|---------------------|---------------------|
    | 이분법      | 선형     | 항상 수렴, 안정적    | 느림, 구간 필요      |
    | 뉴턴-랩슨   | 2차      | 매우 빠름           | 도함수 필요, 발산 가능|
    | 할선법      | ~1.618차 | 도함수 불필요        | 뉴턴보다 느림        |
    | 고정점반복  | 선형~2차 | 간단                | 수렴 조건 확인 필요   |
    | Brent      | 조합     | 안정 + 빠름         | 구현 복잡           |

    실무 권장:
    - scipy.optimize.brentq: 안정적인 근 찾기
    - scipy.optimize.newton: 뉴턴-랩슨/할선법
    - scipy.optimize.fsolve: 다변수 방정식
    """)


if __name__ == "__main__":
    main()
