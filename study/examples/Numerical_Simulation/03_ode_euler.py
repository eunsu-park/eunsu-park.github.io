"""
상미분방정식 - 오일러 방법 (Euler Method)
Ordinary Differential Equations - Euler Method

초기값 문제 dy/dx = f(x, y), y(x₀) = y₀ 를 수치적으로 푸는 가장 기본적인 방법입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


# =============================================================================
# 1. 전진 오일러 방법 (Forward Euler Method)
# =============================================================================
def euler_forward(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    전진 오일러 방법 (명시적 오일러)

    y_{n+1} = y_n + h * f(t_n, y_n)

    오차: O(h) - 1차 정확도
    안정성: 조건부 안정

    Args:
        f: dy/dt = f(t, y)
        y0: 초기값 y(t0)
        t_span: (t0, tf) 시간 구간
        h: 시간 간격

    Returns:
        (t 배열, y 배열)
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y


# =============================================================================
# 2. 후진 오일러 방법 (Backward Euler Method)
# =============================================================================
def euler_backward(
    f: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float,
    newton_tol: float = 1e-10,
    max_iter: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    후진 오일러 방법 (암시적 오일러)

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    암시적 방정식을 뉴턴법으로 해결
    오차: O(h) - 1차 정확도
    안정성: 무조건 안정 (stiff 문제에 적합)

    Args:
        f: dy/dt = f(t, y)
        df_dy: ∂f/∂y (야코비안)
        y0: 초기값
        t_span: 시간 구간
        h: 시간 간격
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        # 뉴턴법으로 y_{n+1} 구하기
        # g(y) = y - y_n - h*f(t_{n+1}, y) = 0
        y_new = y[i]  # 초기 추정값 (전진 오일러)
        t_new = t[i + 1]

        for _ in range(max_iter):
            g = y_new - y[i] - h * f(t_new, y_new)
            dg = 1 - h * df_dy(t_new, y_new)

            if abs(dg) < 1e-15:
                break

            delta = g / dg
            y_new = y_new - delta

            if abs(delta) < newton_tol:
                break

        y[i + 1] = y_new

    return t, y


# =============================================================================
# 3. 수정 오일러 방법 (Modified Euler / Heun's Method)
# =============================================================================
def euler_modified(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    수정 오일러 방법 (Heun's Method)

    예측: y*_{n+1} = y_n + h * f(t_n, y_n)
    수정: y_{n+1} = y_n + h/2 * [f(t_n, y_n) + f(t_{n+1}, y*_{n+1})]

    오차: O(h²) - 2차 정확도
    RK2의 일종
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        y_pred = y[i] + h * k1
        k2 = f(t[i + 1], y_pred)
        y[i + 1] = y[i] + h * (k1 + k2) / 2

    return t, y


# =============================================================================
# 4. 연립 ODE 풀기
# =============================================================================
def euler_system(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    연립 ODE를 전진 오일러로 풀기

    dy/dt = f(t, y), y = [y1, y2, ..., yn]

    Args:
        f: 벡터 함수 f(t, y) -> dy/dt
        y0: 초기값 벡터
        t_span: 시간 구간
        h: 시간 간격

    Returns:
        (t 배열, y 배열) - y.shape = (n_steps+1, n_vars)
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    n_vars = len(y0)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, n_vars))
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = y[i] + h * np.array(f(t[i], y[i]))

    return t, y


# =============================================================================
# 5. 2차 ODE를 1차 연립으로 변환
# =============================================================================
def solve_second_order(
    f: Callable[[float, float, float], float],
    y0: float,
    v0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2차 ODE: y'' = f(t, y, y')
    변환: y₁ = y, y₂ = y'
          y₁' = y₂
          y₂' = f(t, y₁, y₂)

    Args:
        f: y'' = f(t, y, y')
        y0: 초기 위치
        v0: 초기 속도
        t_span, h: 시간 구간과 간격

    Returns:
        (t, y, v) - 위치와 속도
    """
    def system(t, state):
        y, v = state
        return np.array([v, f(t, y, v)])

    t, solution = euler_system(system, np.array([y0, v0]), t_span, h)
    return t, solution[:, 0], solution[:, 1]


# =============================================================================
# 오차 분석
# =============================================================================
def analyze_euler_error():
    """오일러 방법의 오차 분석"""
    # dy/dt = y, y(0) = 1  →  y = e^t
    f = lambda t, y: y
    df_dy = lambda t, y: 1
    y0 = 1
    t_span = (0, 1)
    exact = lambda t: np.exp(t)

    print("\n오일러 방법 오차 분석 (dy/dt = y, y(0) = 1)")
    print("-" * 70)
    print(f"{'h':>10} | {'전진 오일러':>15} | {'수정 오일러':>15} | {'후진 오일러':>15}")
    print("-" * 70)

    hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    for h in hs:
        t1, y1 = euler_forward(f, y0, t_span, h)
        t2, y2 = euler_modified(f, y0, t_span, h)
        t3, y3 = euler_backward(f, df_dy, y0, t_span, h)

        error1 = abs(y1[-1] - exact(1))
        error2 = abs(y2[-1] - exact(1))
        error3 = abs(y3[-1] - exact(1))

        print(f"{h:>10.5f} | {error1:>15.2e} | {error2:>15.2e} | {error3:>15.2e}")


# =============================================================================
# 시각화
# =============================================================================
def plot_euler_comparison():
    """오일러 방법 비교 시각화"""
    # dy/dt = -2y + sin(t), y(0) = 1
    f = lambda t, y: -2*y + np.sin(t)
    df_dy = lambda t, y: -2
    y0 = 1
    t_span = (0, 5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 정확해 (scipy 사용)
    from scipy.integrate import odeint
    t_exact = np.linspace(0, 5, 500)
    y_exact = odeint(lambda y, t: f(t, y), y0, t_exact).flatten()

    # 다양한 h 값 비교
    hs = [0.5, 0.25, 0.1]
    colors = ['r', 'g', 'b']

    ax = axes[0, 0]
    ax.plot(t_exact, y_exact, 'k-', linewidth=2, label='정확해')
    for h, c in zip(hs, colors):
        t, y = euler_forward(f, y0, t_span, h)
        ax.plot(t, y, f'{c}o-', markersize=4, label=f'h={h}')
    ax.set_title('전진 오일러')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_exact, y_exact, 'k-', linewidth=2, label='정확해')
    for h, c in zip(hs, colors):
        t, y = euler_modified(f, y0, t_span, h)
        ax.plot(t, y, f'{c}o-', markersize=4, label=f'h={h}')
    ax.set_title('수정 오일러')
    ax.legend()
    ax.grid(True)

    # 안정성 비교 (stiff 문제)
    # dy/dt = -50(y - cos(t)), y(0) = 0
    f_stiff = lambda t, y: -50*(y - np.cos(t))
    df_stiff = lambda t, y: -50
    h = 0.05
    t_span_stiff = (0, 1)

    ax = axes[1, 0]
    t_ex = np.linspace(0, 1, 500)
    y_ex = odeint(lambda y, t: f_stiff(t, y), 0, t_ex).flatten()
    ax.plot(t_ex, y_ex, 'k-', linewidth=2, label='정확해')

    t1, y1 = euler_forward(f_stiff, 0, t_span_stiff, h)
    t2, y2 = euler_backward(f_stiff, df_stiff, 0, t_span_stiff, h)

    ax.plot(t1, y1, 'r.-', label=f'전진 오일러 (h={h})')
    ax.plot(t2, y2, 'b.-', label=f'후진 오일러 (h={h})')
    ax.set_title('Stiff 문제 안정성')
    ax.legend()
    ax.grid(True)

    # 조화 진동자
    # y'' = -y  →  y' = v, v' = -y
    def harmonic(t, state):
        y, v = state
        return np.array([v, -y])

    ax = axes[1, 1]
    t_ho = np.linspace(0, 20, 1000)
    y_ho_exact = np.cos(t_ho)  # y(0)=1, y'(0)=0

    t, sol = euler_system(harmonic, np.array([1, 0]), (0, 20), 0.1)
    ax.plot(t_ho, y_ho_exact, 'k-', linewidth=2, label='정확해')
    ax.plot(t, sol[:, 0], 'r-', label='오일러 (h=0.1)')
    ax.set_title('조화 진동자 y\'\' = -y')
    ax.set_xlabel('t')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/ode_euler.png', dpi=150)
    plt.close()
    print("그래프 저장: ode_euler.png")


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("상미분방정식 - 오일러 방법")
    print("=" * 60)

    # 예제 1: 지수 감쇠 dy/dt = -y
    print("\n[예제 1] 지수 감쇠: dy/dt = -y, y(0) = 1")
    print("-" * 40)

    f = lambda t, y: -y
    exact = lambda t: np.exp(-t)

    t, y_forward = euler_forward(f, 1, (0, 2), 0.2)
    t, y_modified = euler_modified(f, 1, (0, 2), 0.2)

    print(f"t=2에서:")
    print(f"  정확값:       {exact(2):.6f}")
    print(f"  전진 오일러:  {y_forward[-1]:.6f}")
    print(f"  수정 오일러:  {y_modified[-1]:.6f}")

    # 예제 2: 로지스틱 방정식
    print("\n[예제 2] 로지스틱 방정식: dy/dt = y(1-y), y(0) = 0.1")
    print("-" * 40)

    f_logistic = lambda t, y: y * (1 - y)
    exact_logistic = lambda t: 0.1 * np.exp(t) / (1 + 0.1 * (np.exp(t) - 1))

    t, y = euler_modified(f_logistic, 0.1, (0, 5), 0.1)
    print(f"t=5에서:")
    print(f"  정확값:       {exact_logistic(5):.6f}")
    print(f"  수정 오일러:  {y[-1]:.6f}")

    # 예제 3: 진자 운동 (2차 ODE)
    print("\n[예제 3] 단진자: θ'' = -sin(θ), θ(0) = π/4, θ'(0) = 0")
    print("-" * 40)

    f_pendulum = lambda t, theta, omega: -np.sin(theta)
    t, theta, omega = solve_second_order(f_pendulum, np.pi/4, 0, (0, 10), 0.01)

    print(f"주기적 운동 시뮬레이션 완료")
    print(f"  초기 각도: {np.degrees(np.pi/4):.1f}°")
    print(f"  t=10에서 각도: {np.degrees(theta[-1]):.2f}°")

    # 예제 4: Lotka-Volterra (포식자-피식자 모델)
    print("\n[예제 4] Lotka-Volterra: 포식자-피식자 모델")
    print("-" * 40)

    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0

    def lotka_volterra(t, state):
        x, y = state  # x=피식자, y=포식자
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return np.array([dx, dy])

    t, sol = euler_system(lotka_volterra, np.array([10, 5]), (0, 15), 0.001)
    print(f"  초기: 피식자={10}, 포식자={5}")
    print(f"  t=15: 피식자={sol[-1, 0]:.2f}, 포식자={sol[-1, 1]:.2f}")

    # 오차 분석
    analyze_euler_error()

    # 시각화
    try:
        plot_euler_comparison()
    except Exception as e:
        print(f"그래프 생성 실패: {e}")

    print("\n" + "=" * 60)
    print("오일러 방법 정리")
    print("=" * 60)
    print("""
    | 방법        | 정확도 | 안정성    | 특징                    |
    |------------|--------|----------|-------------------------|
    | 전진 오일러 | O(h)   | 조건부   | 가장 단순, 명시적        |
    | 후진 오일러 | O(h)   | 무조건   | 암시적, stiff에 적합     |
    | 수정 오일러 | O(h²)  | 조건부   | 2차 정확도, RK2의 일종   |

    한계:
    - 정확도가 낮음 (더 높은 차수 필요시 RK4 사용)
    - 에너지 보존 불가 (심플렉틱 적분기 필요)

    실무:
    - scipy.integrate.odeint: 적응적 다단계 방법
    - scipy.integrate.solve_ivp: 다양한 방법 선택 가능
    """)


if __name__ == "__main__":
    main()
