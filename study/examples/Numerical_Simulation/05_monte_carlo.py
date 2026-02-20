"""
몬테카를로 시뮬레이션 (Monte Carlo Simulation)
Monte Carlo Methods

확률적 방법을 사용한 수치 계산 및 시뮬레이션입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# =============================================================================
# 1. π 추정 (원의 넓이)
# =============================================================================
def estimate_pi(n_samples: int) -> Tuple[float, float]:
    """
    단위 원을 사용한 π 추정

    정사각형 [-1, 1] x [-1, 1] 내에 무작위 점을 뿌리고
    단위 원 내부에 떨어지는 비율을 계산

    원의 넓이 / 정사각형 넓이 = π / 4
    → π = 4 * (원 내부 점 / 전체 점)

    오차: O(1/√n)
    """
    # 균일 분포로 점 생성
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)

    # 원 내부에 있는 점 개수
    inside = np.sum(x**2 + y**2 <= 1)

    pi_estimate = 4 * inside / n_samples
    std_error = 4 * np.sqrt(inside * (n_samples - inside) / n_samples**3)

    return pi_estimate, std_error


def estimate_pi_convergence(max_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    """π 추정의 수렴 과정"""
    x = np.random.uniform(-1, 1, max_samples)
    y = np.random.uniform(-1, 1, max_samples)
    inside = (x**2 + y**2 <= 1).cumsum()
    n = np.arange(1, max_samples + 1)
    return n, 4 * inside / n


# =============================================================================
# 2. 몬테카를로 적분
# =============================================================================
def monte_carlo_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_samples: int
) -> Tuple[float, float]:
    """
    1D 몬테카를로 적분

    ∫[a,b] f(x)dx ≈ (b-a) * (1/n) * Σf(x_i)

    Args:
        f: 피적분 함수
        a, b: 적분 구간
        n_samples: 샘플 수

    Returns:
        (적분값 추정, 표준 오차)
    """
    x = np.random.uniform(a, b, n_samples)
    fx = f(x)

    integral = (b - a) * np.mean(fx)
    variance = np.var(fx)
    std_error = (b - a) * np.sqrt(variance / n_samples)

    return integral, std_error


def monte_carlo_integrate_nd(
    f: Callable[[np.ndarray], float],
    bounds: list,
    n_samples: int
) -> Tuple[float, float]:
    """
    다차원 몬테카를로 적분

    Args:
        f: 다변수 함수 f(x) where x is array
        bounds: [(a1,b1), (a2,b2), ...] 각 차원의 범위
        n_samples: 샘플 수
    """
    dim = len(bounds)
    volume = np.prod([b - a for a, b in bounds])

    # 각 차원에서 균일 샘플링
    samples = np.array([
        np.random.uniform(a, b, n_samples)
        for a, b in bounds
    ]).T  # shape: (n_samples, dim)

    values = np.array([f(x) for x in samples])

    integral = volume * np.mean(values)
    std_error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, std_error


# =============================================================================
# 3. 중요도 샘플링 (Importance Sampling)
# =============================================================================
def importance_sampling(
    f: Callable[[np.ndarray], np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    sample_g: Callable[[int], np.ndarray],
    n_samples: int
) -> Tuple[float, float]:
    """
    중요도 샘플링

    목표: ∫f(x)dx 를 더 효율적으로 추정

    g(x)를 중요도 분포로 사용:
    ∫f(x)dx = ∫(f(x)/g(x))g(x)dx = E_g[f(x)/g(x)]

    분산 감소: f(x)와 비슷한 g(x) 선택

    Args:
        f: 피적분 함수 * 원래 분포
        g: 중요도 분포의 PDF
        sample_g: g에서 샘플 생성 함수
        n_samples: 샘플 수
    """
    x = sample_g(n_samples)
    weights = f(x) / g(x)

    integral = np.mean(weights)
    std_error = np.std(weights) / np.sqrt(n_samples)

    return integral, std_error


# =============================================================================
# 4. 랜덤 워크 시뮬레이션
# =============================================================================
def random_walk_1d(n_steps: int, n_walks: int = 1) -> np.ndarray:
    """
    1D 랜덤 워크

    각 스텝에서 +1 또는 -1로 이동
    """
    steps = np.random.choice([-1, 1], size=(n_walks, n_steps))
    positions = np.cumsum(steps, axis=1)
    return positions


def random_walk_2d(n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """2D 랜덤 워크"""
    angles = np.random.uniform(0, 2*np.pi, n_steps)
    dx = np.cos(angles)
    dy = np.sin(angles)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y


# =============================================================================
# 5. 옵션 가격 결정 (Black-Scholes Monte Carlo)
# =============================================================================
def black_scholes_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int,
    option_type: str = 'call'
) -> Tuple[float, float]:
    """
    유럽형 옵션 가격의 몬테카를로 추정

    기하 브라운 운동:
    S_T = S_0 * exp((r - σ²/2)T + σ√T * Z)

    Args:
        S0: 현재 주가
        K: 행사가
        T: 만기 (년)
        r: 무위험 이자율
        sigma: 변동성
        n_simulations: 시뮬레이션 횟수
        option_type: 'call' or 'put'
    """
    # 주가 시뮬레이션
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # 페이오프 계산
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:  # put
        payoffs = np.maximum(K - ST, 0)

    # 현재 가치로 할인
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

    return price, std_error


def black_scholes_analytical(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """Black-Scholes 해석적 해 (비교용)"""
    from scipy.stats import norm

    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return price


# =============================================================================
# 6. 버핏 바늘 문제 (Buffon's Needle)
# =============================================================================
def buffon_needle(
    needle_length: float,
    line_spacing: float,
    n_drops: int
) -> Tuple[float, float]:
    """
    버핏 바늘 문제로 π 추정

    평행선 사이에 바늘을 떨어뜨렸을 때
    선을 교차할 확률 = 2L / (πD)

    → π = 2L * n_drops / (D * crossings)

    Args:
        needle_length: 바늘 길이 (L)
        line_spacing: 선 간격 (D), L ≤ D
        n_drops: 바늘 떨어뜨리기 횟수
    """
    if needle_length > line_spacing:
        raise ValueError("바늘 길이는 선 간격보다 작아야 합니다")

    # 바늘 중심의 위치 (0 ~ D/2)
    y_center = np.random.uniform(0, line_spacing/2, n_drops)

    # 바늘 각도 (0 ~ π)
    theta = np.random.uniform(0, np.pi, n_drops)

    # 바늘 끝이 선을 넘는지 확인
    # 바늘 끝의 y 좌표 변화: (L/2) * sin(θ)
    crosses = y_center <= (needle_length/2) * np.sin(theta)
    n_crossings = np.sum(crosses)

    if n_crossings == 0:
        return float('inf'), float('inf')

    pi_estimate = 2 * needle_length * n_drops / (line_spacing * n_crossings)

    return pi_estimate, 1 / np.sqrt(n_crossings)  # 대략적인 오차


# =============================================================================
# 시각화
# =============================================================================
def plot_monte_carlo_examples():
    """몬테카를로 예제 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. π 추정
    ax = axes[0, 0]
    n = 10000
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = x**2 + y**2 <= 1

    ax.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5)
    ax.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    pi_est = 4 * np.sum(inside) / n
    ax.set_title(f'π 추정: {pi_est:.4f} (n={n})')

    # 2. π 추정 수렴
    ax = axes[0, 1]
    n, estimates = estimate_pi_convergence(50000)
    ax.semilogx(n, estimates, 'b-', alpha=0.7)
    ax.axhline(y=np.pi, color='r', linestyle='--', label=f'π = {np.pi:.6f}')
    ax.set_xlabel('샘플 수')
    ax.set_ylabel('π 추정값')
    ax.set_title('π 추정 수렴')
    ax.legend()
    ax.grid(True)

    # 3. 랜덤 워크
    ax = axes[1, 0]
    for _ in range(5):
        x, y = random_walk_2d(1000)
        ax.plot(np.concatenate([[0], x]), np.concatenate([[0], y]), alpha=0.7)
    ax.plot(0, 0, 'ko', markersize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D 랜덤 워크 (5개 경로)')
    ax.axis('equal')
    ax.grid(True)

    # 4. 옵션 가격 수렴
    ax = axes[1, 1]
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    exact = black_scholes_analytical(S0, K, T, r, sigma)

    ns = np.logspace(2, 5, 20).astype(int)
    estimates = []
    errors = []
    for n in ns:
        est, err = black_scholes_mc(S0, K, T, r, sigma, n)
        estimates.append(est)
        errors.append(err)

    ax.errorbar(ns, estimates, yerr=errors, fmt='o-', capsize=3)
    ax.axhline(y=exact, color='r', linestyle='--', label=f'해석해: {exact:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('시뮬레이션 횟수')
    ax.set_ylabel('옵션 가격')
    ax.set_title('콜옵션 가격 수렴')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/monte_carlo.png', dpi=150)
    plt.close()
    print("그래프 저장: monte_carlo.png")


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("몬테카를로 시뮬레이션 (Monte Carlo)")
    print("=" * 60)

    np.random.seed(42)

    # 1. π 추정
    print("\n[1] π 추정 (원의 넓이)")
    print("-" * 40)
    for n in [1000, 10000, 100000, 1000000]:
        pi_est, std_err = estimate_pi(n)
        print(f"n={n:>8}: π ≈ {pi_est:.6f} ± {std_err:.6f}, "
              f"오차: {abs(pi_est - np.pi):.6f}")

    # 2. 몬테카를로 적분
    print("\n[2] 몬테카를로 적분")
    print("-" * 40)

    # ∫[0,1] x² dx = 1/3
    f1 = lambda x: x**2
    integral, error = monte_carlo_integrate(f1, 0, 1, 100000)
    print(f"∫x²dx [0,1] = {integral:.6f} ± {error:.6f} (정확: 0.333333)")

    # ∫[0,π] sin(x) dx = 2
    f2 = lambda x: np.sin(x)
    integral, error = monte_carlo_integrate(f2, 0, np.pi, 100000)
    print(f"∫sin(x)dx [0,π] = {integral:.6f} ± {error:.6f} (정확: 2.0)")

    # 3. 다차원 적분
    print("\n[3] 다차원 적분")
    print("-" * 40)

    # 단위 구의 부피: 4π/3 ≈ 4.189
    def sphere_indicator(x):
        return 1 if np.sum(x**2) <= 1 else 0

    volume, error = monte_carlo_integrate_nd(sphere_indicator, [(-1,1)]*3, 100000)
    print(f"단위 구 부피 = {volume:.4f} ± {error:.4f} (정확: {4*np.pi/3:.4f})")

    # 4. 옵션 가격
    print("\n[4] 유럽형 콜옵션 가격")
    print("-" * 40)

    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    exact = black_scholes_analytical(S0, K, T, r, sigma)
    mc_price, mc_error = black_scholes_mc(S0, K, T, r, sigma, 100000)

    print(f"Black-Scholes 해석해: {exact:.4f}")
    print(f"Monte Carlo (n=100000): {mc_price:.4f} ± {mc_error:.4f}")
    print(f"오차: {abs(mc_price - exact):.4f}")

    # 5. 버핏 바늘
    print("\n[5] 버핏 바늘 문제")
    print("-" * 40)

    pi_est, _ = buffon_needle(1, 2, 100000)
    print(f"π 추정 (L=1, D=2, n=100000): {pi_est:.6f}")

    # 6. 랜덤 워크 통계
    print("\n[6] 1D 랜덤 워크 통계")
    print("-" * 40)

    n_steps = 1000
    n_walks = 10000
    positions = random_walk_1d(n_steps, n_walks)
    final_positions = positions[:, -1]

    print(f"{n_steps}스텝 후 ({n_walks}개 워크):")
    print(f"  평균 위치: {np.mean(final_positions):.2f} (이론: 0)")
    print(f"  표준편차: {np.std(final_positions):.2f} (이론: {np.sqrt(n_steps):.2f})")

    # 시각화
    try:
        plot_monte_carlo_examples()
    except Exception as e:
        print(f"그래프 생성 실패: {e}")

    print("\n" + "=" * 60)
    print("몬테카를로 방법 정리")
    print("=" * 60)
    print("""
    장점:
    - 고차원 문제에서도 수렴 속도 유지 (차원의 저주 회피)
    - 복잡한 영역/조건에 적용 용이
    - 구현이 단순

    단점:
    - 수렴 속도가 느림: O(1/√n)
    - 확률적 → 결과에 불확실성
    - 많은 샘플 필요

    분산 감소 기법:
    - 중요도 샘플링 (Importance Sampling)
    - 대조 변량 (Control Variates)
    - 안티테틱 변량 (Antithetic Variates)
    - 층화 샘플링 (Stratified Sampling)
    - 준난수 (Quasi-random / Low-discrepancy sequences)

    응용:
    - 금융: 파생상품 가격 결정, 리스크 분석
    - 물리: 통계역학, 입자 시뮬레이션
    - 컴퓨터 그래픽: 경로 추적 렌더링
    - 최적화: 시뮬레이티드 어닐링, MCMC
    """)


if __name__ == "__main__":
    main()
