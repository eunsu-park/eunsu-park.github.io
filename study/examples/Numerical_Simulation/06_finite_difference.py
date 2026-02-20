"""
유한 차분법 (Finite Difference Method)
Finite Difference Methods for PDEs

편미분방정식(PDE)을 수치적으로 푸는 방법입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# =============================================================================
# 1. 1D 열 방정식 (Heat Equation)
# =============================================================================
def heat_equation_explicit(
    L: float,
    T: float,
    nx: int,
    nt: int,
    alpha: float,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_left: float = 0,
    boundary_right: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D 열 방정식 (명시적 방법)

    ∂u/∂t = α ∂²u/∂x²

    FTCS (Forward Time, Central Space):
    u(i,n+1) = u(i,n) + r[u(i+1,n) - 2u(i,n) + u(i-1,n)]
    여기서 r = α*dt/dx²

    안정성 조건: r ≤ 0.5

    Args:
        L: 공간 영역 [0, L]
        T: 시간 영역 [0, T]
        nx: 공간 격자 수
        nt: 시간 스텝 수
        alpha: 열확산 계수
        initial_condition: 초기 조건 함수 u(x, 0)
        boundary_left, boundary_right: 경계 조건

    Returns:
        (x 배열, t 배열, u 배열)
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2

    print(f"dx={dx:.4f}, dt={dt:.6f}, r={r:.4f}")
    if r > 0.5:
        print(f"경고: 안정성 조건 위반 (r={r:.4f} > 0.5)")

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))

    # 초기 조건
    u[0, :] = initial_condition(x)

    # 경계 조건
    u[:, 0] = boundary_left
    u[:, -1] = boundary_right

    # 시간 전진
    for n in range(nt):
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

    return x, t, u


def heat_equation_implicit(
    L: float,
    T: float,
    nx: int,
    nt: int,
    alpha: float,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_left: float = 0,
    boundary_right: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D 열 방정식 (암시적 방법 - Crank-Nicolson)

    무조건 안정, O(dt², dx²) 정확도
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / (2 * dx**2)

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))

    # 초기 조건
    u[0, :] = initial_condition(x)
    u[:, 0] = boundary_left
    u[:, -1] = boundary_right

    # 삼대각 행렬 설정
    n_inner = nx - 2
    A = np.zeros((n_inner, n_inner))
    B = np.zeros((n_inner, n_inner))

    for i in range(n_inner):
        A[i, i] = 1 + 2 * r
        B[i, i] = 1 - 2 * r
        if i > 0:
            A[i, i - 1] = -r
            B[i, i - 1] = r
        if i < n_inner - 1:
            A[i, i + 1] = -r
            B[i, i + 1] = r

    # 시간 전진
    for n in range(nt):
        # 우변 계산
        b = B @ u[n, 1:-1]
        b[0] += r * (u[n + 1, 0] + u[n, 0])
        b[-1] += r * (u[n + 1, -1] + u[n, -1])

        # 선형 시스템 풀기
        u[n + 1, 1:-1] = np.linalg.solve(A, b)

    return x, t, u


# =============================================================================
# 2. 1D 파동 방정식 (Wave Equation)
# =============================================================================
def wave_equation(
    L: float,
    T: float,
    nx: int,
    nt: int,
    c: float,
    initial_displacement: Callable[[np.ndarray], np.ndarray],
    initial_velocity: Callable[[np.ndarray], np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D 파동 방정식

    ∂²u/∂t² = c² ∂²u/∂x²

    유한 차분:
    u(i,n+1) = 2u(i,n) - u(i,n-1) + s²[u(i+1,n) - 2u(i,n) + u(i-1,n)]
    여기서 s = c*dt/dx (Courant number)

    안정성 조건: s ≤ 1 (CFL 조건)
    """
    dx = L / (nx - 1)
    dt = T / nt
    s = c * dt / dx

    print(f"Courant number s={s:.4f}")
    if s > 1:
        print(f"경고: CFL 조건 위반 (s={s:.4f} > 1)")

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt + 1)
    u = np.zeros((nt + 1, nx))

    # 초기 조건
    u[0, :] = initial_displacement(x)

    # 첫 번째 시간 스텝 (초기 속도 사용)
    if initial_velocity is None:
        initial_velocity = lambda x: np.zeros_like(x)

    v0 = initial_velocity(x)
    for i in range(1, nx - 1):
        u[1, i] = (u[0, i] + dt * v0[i] +
                   0.5 * s**2 * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1]))

    # 경계 조건 (고정)
    u[:, 0] = 0
    u[:, -1] = 0

    # 시간 전진
    for n in range(1, nt):
        for i in range(1, nx - 1):
            u[n + 1, i] = (2 * u[n, i] - u[n - 1, i] +
                          s**2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]))

    return x, t, u


# =============================================================================
# 3. 2D 라플라스/포아송 방정식
# =============================================================================
def laplace_2d(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    boundary_conditions: dict,
    max_iter: int = 10000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D 라플라스 방정식 (Jacobi 반복법)

    ∇²u = 0  또는  ∂²u/∂x² + ∂²u/∂y² = 0

    Jacobi 반복:
    u(i,j)_new = 0.25 * [u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1)]

    Args:
        boundary_conditions: {'top': val, 'bottom': val, 'left': val, 'right': val}
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    u = np.zeros((ny, nx))

    # 경계 조건 설정
    bc = boundary_conditions
    if callable(bc.get('top')):
        u[-1, :] = bc['top'](x)
    else:
        u[-1, :] = bc.get('top', 0)

    if callable(bc.get('bottom')):
        u[0, :] = bc['bottom'](x)
    else:
        u[0, :] = bc.get('bottom', 0)

    if callable(bc.get('left')):
        u[:, 0] = bc['left'](y)
    else:
        u[:, 0] = bc.get('left', 0)

    if callable(bc.get('right')):
        u[:, -1] = bc['right'](y)
    else:
        u[:, -1] = bc.get('right', 0)

    # Jacobi 반복
    for iteration in range(max_iter):
        u_old = u.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = 0.25 * (u_old[i + 1, j] + u_old[i - 1, j] +
                                  u_old[i, j + 1] + u_old[i, j - 1])

        # 수렴 체크
        error = np.max(np.abs(u - u_old))
        if error < tol:
            print(f"수렴: {iteration + 1}회 반복, 오차={error:.2e}")
            break

    return x, y, u


def laplace_2d_sor(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    boundary_conditions: dict,
    omega: float = 1.5,
    max_iter: int = 10000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D 라플라스 방정식 (SOR - Successive Over-Relaxation)

    Jacobi보다 빠른 수렴

    Args:
        omega: 이완 인자 (1 < ω < 2 for 가속)
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    u = np.zeros((ny, nx))

    bc = boundary_conditions
    u[-1, :] = bc.get('top', 0) if not callable(bc.get('top')) else bc['top'](x)
    u[0, :] = bc.get('bottom', 0) if not callable(bc.get('bottom')) else bc['bottom'](x)
    u[:, 0] = bc.get('left', 0) if not callable(bc.get('left')) else bc['left'](y)
    u[:, -1] = bc.get('right', 0) if not callable(bc.get('right')) else bc['right'](y)

    for iteration in range(max_iter):
        max_diff = 0

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u_old = u[i, j]
                u_gs = 0.25 * (u[i + 1, j] + u[i - 1, j] +
                              u[i, j + 1] + u[i, j - 1])
                u[i, j] = (1 - omega) * u_old + omega * u_gs
                max_diff = max(max_diff, abs(u[i, j] - u_old))

        if max_diff < tol:
            print(f"SOR 수렴: {iteration + 1}회 반복")
            break

    return x, y, u


# =============================================================================
# 시각화
# =============================================================================
def plot_heat_equation():
    """열 방정식 시각화"""
    # 초기 조건: 가운데 뜨거운 부분
    initial = lambda x: np.sin(np.pi * x)

    x, t, u = heat_equation_explicit(
        L=1.0, T=0.5, nx=51, nt=500, alpha=0.01,
        initial_condition=initial
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 시공간 등고선
    ax = axes[0]
    X, T_mesh = np.meshgrid(x, t)
    contour = ax.contourf(X, T_mesh, u, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax, label='온도')
    ax.set_xlabel('위치 x')
    ax.set_ylabel('시간 t')
    ax.set_title('열 방정식 해')

    # 시간별 프로파일
    ax = axes[1]
    times_to_plot = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    for idx in times_to_plot:
        ax.plot(x, u[idx, :], label=f't={t[idx]:.3f}')
    ax.set_xlabel('위치 x')
    ax.set_ylabel('온도 u')
    ax.set_title('시간별 온도 분포')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/heat_equation.png', dpi=150)
    plt.close()
    print("그래프 저장: heat_equation.png")


def plot_wave_equation():
    """파동 방정식 시각화"""
    # 초기 변위: 가우시안 펄스
    initial = lambda x: np.exp(-100 * (x - 0.5)**2)

    x, t, u = wave_equation(
        L=1.0, T=2.0, nx=101, nt=400, c=1.0,
        initial_displacement=initial
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 시공간 등고선
    ax = axes[0]
    X, T_mesh = np.meshgrid(x, t)
    contour = ax.contourf(X, T_mesh, u, levels=20, cmap='RdBu')
    plt.colorbar(contour, ax=ax, label='변위')
    ax.set_xlabel('위치 x')
    ax.set_ylabel('시간 t')
    ax.set_title('파동 방정식 해')

    # 스냅샷
    ax = axes[1]
    for idx in [0, 50, 100, 150, 200]:
        ax.plot(x, u[idx, :], label=f't={t[idx]:.2f}')
    ax.set_xlabel('위치 x')
    ax.set_ylabel('변위 u')
    ax.set_title('시간별 파동 형태')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/wave_equation.png', dpi=150)
    plt.close()
    print("그래프 저장: wave_equation.png")


def plot_laplace_2d():
    """2D 라플라스 방정식 시각화"""
    bc = {
        'top': lambda x: np.sin(np.pi * x),
        'bottom': 0,
        'left': 0,
        'right': 0
    }

    x, y, u = laplace_2d(
        Lx=1.0, Ly=1.0, nx=51, ny=51,
        boundary_conditions=bc,
        tol=1e-5
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    X, Y = np.meshgrid(x, y)

    # 등고선
    ax = axes[0]
    contour = ax.contourf(X, Y, u, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('라플라스 방정식 해')
    ax.set_aspect('equal')

    # 3D 표면
    ax = axes[1]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, u, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title('3D 표면')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/laplace_2d.png', dpi=150)
    plt.close()
    print("그래프 저장: laplace_2d.png")


# =============================================================================
# 테스트
# =============================================================================
def main():
    print("=" * 60)
    print("유한 차분법 (Finite Difference Method)")
    print("=" * 60)

    # 1. 열 방정식
    print("\n[1] 1D 열 방정식")
    print("-" * 40)

    initial = lambda x: np.sin(np.pi * x)

    print("명시적 방법 (FTCS):")
    x, t, u_explicit = heat_equation_explicit(
        L=1.0, T=0.1, nx=21, nt=100, alpha=0.1,
        initial_condition=initial
    )
    print(f"  t=0.1에서 중앙 온도: {u_explicit[-1, 10]:.6f}")

    print("\n암시적 방법 (Crank-Nicolson):")
    x, t, u_implicit = heat_equation_implicit(
        L=1.0, T=0.1, nx=21, nt=100, alpha=0.1,
        initial_condition=initial
    )
    print(f"  t=0.1에서 중앙 온도: {u_implicit[-1, 10]:.6f}")

    # 해석해: u = exp(-π²αt) sin(πx)
    exact = np.exp(-np.pi**2 * 0.1 * 0.1) * np.sin(np.pi * 0.5)
    print(f"  해석해: {exact:.6f}")

    # 2. 파동 방정식
    print("\n[2] 1D 파동 방정식")
    print("-" * 40)

    initial_wave = lambda x: np.sin(np.pi * x)

    x, t, u_wave = wave_equation(
        L=1.0, T=2.0, nx=51, nt=200, c=1.0,
        initial_displacement=initial_wave
    )
    print(f"주기 T = 2L/c = 2.0")
    print(f"t=2.0에서 중앙 변위: {u_wave[-1, 25]:.6f}")
    print(f"(초기값과 같아야 함: {initial_wave(0.5):.6f})")

    # 3. 2D 라플라스 방정식
    print("\n[3] 2D 라플라스 방정식")
    print("-" * 40)

    bc = {'top': 100, 'bottom': 0, 'left': 0, 'right': 0}

    print("Jacobi 반복:")
    x, y, u_jacobi = laplace_2d(1.0, 1.0, 31, 31, bc, tol=1e-4)

    print("\nSOR (ω=1.5):")
    x, y, u_sor = laplace_2d_sor(1.0, 1.0, 31, 31, bc, omega=1.5, tol=1e-4)

    print(f"\n중심점 온도: {u_jacobi[15, 15]:.4f}")

    # 시각화
    try:
        plot_heat_equation()
        plot_wave_equation()
        plot_laplace_2d()
    except Exception as e:
        print(f"그래프 생성 실패: {e}")

    print("\n" + "=" * 60)
    print("유한 차분법 정리")
    print("=" * 60)
    print("""
    PDE 유형과 방법:

    | PDE 유형    | 대표 예      | 권장 방법              |
    |------------|-------------|----------------------|
    | 포물선형    | 열 방정식    | FTCS, Crank-Nicolson |
    | 쌍곡선형    | 파동 방정식  | 중심차분, Lax-Wendroff|
    | 타원형      | 라플라스     | Jacobi, GS, SOR      |

    안정성 조건:
    - 열 방정식 (명시적): r = αΔt/Δx² ≤ 0.5
    - 파동 방정식: CFL = cΔt/Δx ≤ 1

    차분 근사:
    - 전진 차분: ∂u/∂t ≈ [u(t+Δt) - u(t)] / Δt
    - 후진 차분: ∂u/∂t ≈ [u(t) - u(t-Δt)] / Δt
    - 중심 차분: ∂²u/∂x² ≈ [u(x+Δx) - 2u(x) + u(x-Δx)] / Δx²

    실무:
    - scipy.ndimage: 간단한 필터링
    - FEniCS, FiPy: Python PDE 프레임워크
    - OpenFOAM: CFD (유한 체적법)
    """)


if __name__ == "__main__":
    main()
