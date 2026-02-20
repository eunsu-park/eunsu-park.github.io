"""
동적 프로그래밍 (Dynamic Programming) 구현
- Policy Evaluation (정책 평가)
- Policy Improvement (정책 개선)
- Policy Iteration (정책 반복)
- Value Iteration (가치 반복)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict


class GridWorld:
    """간단한 그리드 월드 환경"""

    def __init__(self, size=4):
        self.size = size
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)

    def get_states(self):
        """모든 상태 반환"""
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def is_terminal(self, state):
        """종료 상태 확인"""
        return state == (0, 0) or state == (self.size-1, self.size-1)

    def get_transitions(self, state, action):
        """전이 확률 반환: [(prob, next_state, reward, done)]"""
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        delta = deltas[action]

        # 그리드 경계 처리
        new_row = max(0, min(self.size-1, state[0] + delta[0]))
        new_col = max(0, min(self.size-1, state[1] + delta[1]))
        next_state = (new_row, new_col)

        # 보상: 각 이동마다 -1
        reward = -1
        done = self.is_terminal(next_state)

        return [(1.0, next_state, reward, done)]


def create_uniform_policy(grid):
    """균등 랜덤 정책 생성"""
    policy = {}
    for s in grid.get_states():
        policy[s] = {a: 1.0/len(grid.actions) for a in grid.actions}
    return policy


def policy_evaluation(grid, policy: Dict, gamma: float = 0.9, theta: float = 1e-6):
    """
    정책 평가: 주어진 정책의 가치 함수 계산

    Args:
        grid: GridWorld 환경
        policy: 정책 {state: {action: probability}}
        gamma: 할인율
        theta: 수렴 임계값

    Returns:
        V: 상태 가치 함수 {state: value}
    """
    # 가치 함수 초기화
    V = {s: 0.0 for s in grid.get_states()}

    iteration = 0
    while True:
        delta = 0  # 최대 변화량 추적
        iteration += 1

        # 모든 상태에 대해 업데이트
        for s in grid.get_states():
            if grid.is_terminal(s):
                continue

            v = V[s]  # 이전 값 저장
            new_v = 0

            # 벨만 기대 방정식 적용
            for a in grid.actions:
                action_prob = policy[s].get(a, 0)

                for prob, next_s, reward, done in grid.get_transitions(s, a):
                    if done:
                        new_v += action_prob * prob * reward
                    else:
                        new_v += action_prob * prob * (reward + gamma * V[next_s])

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        # 수렴 체크
        if delta < theta:
            print(f"정책 평가 수렴: {iteration} iterations, delta={delta:.8f}")
            break

    return V


def policy_improvement(grid, V: Dict, gamma: float = 0.9):
    """
    정책 개선: V를 기반으로 탐욕적 정책 생성

    Args:
        grid: GridWorld 환경
        V: 현재 가치 함수
        gamma: 할인율

    Returns:
        new_policy: 개선된 정책
        policy_stable: 정책이 변하지 않았으면 True
    """
    new_policy = {}
    policy_stable = True

    for s in grid.get_states():
        if grid.is_terminal(s):
            new_policy[s] = {a: 1.0/len(grid.actions) for a in grid.actions}
            continue

        # 각 행동의 Q 값 계산
        q_values = {}
        for a in grid.actions:
            q = 0
            for prob, next_s, reward, done in grid.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        # 최적 행동 찾기
        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        # 동률인 행동들 찾기 (수치 오차 고려)
        best_actions = [a for a, q in q_values.items()
                        if abs(q - best_q) < 1e-8]

        # 결정적 정책 생성
        new_policy[s] = {a: 0.0 for a in grid.actions}
        for a in best_actions:
            new_policy[s][a] = 1.0 / len(best_actions)

    return new_policy, policy_stable


def policy_iteration(grid, gamma: float = 0.9, theta: float = 1e-6):
    """
    정책 반복 알고리즘

    Returns:
        V: 최적 가치 함수
        policy: 최적 정책
    """
    # 균등 랜덤 정책으로 초기화
    policy = create_uniform_policy(grid)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== 정책 반복 {iteration} ===")

        # 1. 정책 평가
        V = policy_evaluation(grid, policy, gamma, theta)

        # 2. 정책 개선
        old_policy = policy.copy()
        policy, _ = policy_improvement(grid, V, gamma)

        # 3. 정책 안정성 체크
        policy_stable = True
        for s in grid.get_states():
            if grid.is_terminal(s):
                continue

            old_best = max(old_policy[s], key=old_policy[s].get)
            new_best = max(policy[s], key=policy[s].get)

            if old_best != new_best:
                policy_stable = False
                break

        if policy_stable:
            print(f"\n정책 반복 수렴! (총 {iteration} iterations)")
            break

    return V, policy


def value_iteration(grid, gamma: float = 0.9, theta: float = 1e-6):
    """
    가치 반복 알고리즘

    Returns:
        V: 최적 가치 함수
        policy: 최적 정책
    """
    # 가치 함수 초기화
    V = {s: 0.0 for s in grid.get_states()}

    iteration = 0
    while True:
        delta = 0
        iteration += 1

        for s in grid.get_states():
            if grid.is_terminal(s):
                continue

            v = V[s]

            # 벨만 최적성 방정식: max over actions
            q_values = []
            for a in grid.actions:
                q = 0
                for prob, next_s, reward, done in grid.get_transitions(s, a):
                    if done:
                        q += prob * reward
                    else:
                        q += prob * (reward + gamma * V[next_s])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if iteration % 10 == 0:
            print(f"반복 {iteration}: delta = {delta:.8f}")

        if delta < theta:
            print(f"\n가치 반복 수렴: {iteration} iterations")
            break

    # 최적 정책 추출
    policy = {}
    for s in grid.get_states():
        if grid.is_terminal(s):
            policy[s] = {a: 1.0/len(grid.actions) for a in grid.actions}
            continue

        q_values = {}
        for a in grid.actions:
            q = 0
            for prob, next_s, reward, done in grid.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        best_action = max(q_values, key=q_values.get)
        policy[s] = {a: 0.0 for a in grid.actions}
        policy[s][best_action] = 1.0

    return V, policy


def print_value_function(grid, V):
    """가치 함수 출력"""
    print("\n가치 함수:")
    for i in range(grid.size):
        row = [f"{V[(i,j)]:7.2f}" for j in range(grid.size)]
        print(" ".join(row))


def print_policy(grid, policy):
    """정책 출력 (화살표로)"""
    print("\n최적 정책:")
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

    for i in range(grid.size):
        row = []
        for j in range(grid.size):
            s = (i, j)
            if grid.is_terminal(s):
                row.append('  *  ')
            else:
                best_a = max(policy[s], key=policy[s].get)
                row.append(f'  {arrows[best_a]}  ')
        print(" ".join(row))


def visualize_value_function(grid, V, title="Value Function"):
    """가치 함수 시각화"""
    value_grid = np.zeros((grid.size, grid.size))
    for i in range(grid.size):
        for j in range(grid.size):
            value_grid[i, j] = V[(i, j)]

    plt.figure(figsize=(8, 6))
    plt.imshow(value_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)

    # 숫자 표시
    for i in range(grid.size):
        for j in range(grid.size):
            plt.text(j, i, f'{value_grid[i, j]:.1f}',
                    ha='center', va='center', color='black', fontsize=12)

    plt.xticks(range(grid.size))
    plt.yticks(range(grid.size))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('value_function.png', dpi=150)
    print(f"가치 함수 시각화 저장: value_function.png")


def compare_algorithms():
    """DP 알고리즘 비교"""
    print("=" * 60)
    print("동적 프로그래밍 알고리즘 비교")
    print("=" * 60)

    grid = GridWorld(size=4)
    gamma = 0.9

    # 1. 정책 평가 (균등 랜덤 정책)
    print("\n[1] 정책 평가 - 균등 랜덤 정책")
    print("-" * 60)
    uniform_policy = create_uniform_policy(grid)
    V_uniform = policy_evaluation(grid, uniform_policy, gamma)
    print_value_function(grid, V_uniform)

    # 2. 정책 반복
    print("\n[2] 정책 반복 (Policy Iteration)")
    print("-" * 60)
    V_pi, policy_pi = policy_iteration(grid, gamma)
    print_value_function(grid, V_pi)
    print_policy(grid, policy_pi)

    # 3. 가치 반복
    print("\n[3] 가치 반복 (Value Iteration)")
    print("-" * 60)
    V_vi, policy_vi = value_iteration(grid, gamma)
    print_value_function(grid, V_vi)
    print_policy(grid, policy_vi)

    # 4. 결과 비교
    print("\n[4] 결과 비교")
    print("-" * 60)
    print("정책 반복과 가치 반복의 가치 함수 차이:")
    max_diff = 0
    for s in grid.get_states():
        diff = abs(V_pi[s] - V_vi[s])
        max_diff = max(max_diff, diff)
    print(f"최대 차이: {max_diff:.10f}")

    # 시각화
    visualize_value_function(grid, V_pi, "Policy Iteration - Value Function")

    return V_pi, policy_pi, V_vi, policy_vi


def frozen_lake_example():
    """Frozen Lake 환경에서 DP 적용"""
    import gymnasium as gym

    print("\n" + "=" * 60)
    print("Frozen Lake 예제")
    print("=" * 60)

    # 환경 생성 (미끄러지지 않는 버전)
    env = gym.make('FrozenLake-v1', is_slippery=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    gamma = 0.99
    theta = 1e-8

    # P[s][a] = [(prob, next_state, reward, done), ...]
    P = env.unwrapped.P

    # 가치 반복
    V = np.zeros(n_states)
    iteration = 0

    print("\n가치 반복 시작...")
    while True:
        delta = 0
        iteration += 1

        for s in range(n_states):
            v = V[s]

            # 각 행동의 가치 계산
            q_values = []
            for a in range(n_actions):
                q = sum(prob * (reward + gamma * V[next_s] * (not done))
                       for prob, next_s, reward, done in P[s][a])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            print(f"수렴: {iteration} iterations")
            break

    # 최적 정책 추출
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = sum(prob * (reward + gamma * V[next_s] * (not done))
                   for prob, next_s, reward, done in P[s][a])
            q_values.append(q)
        policy[s] = np.argmax(q_values)

    # 결과 시각화
    action_names = ['←', '↓', '→', '↑']
    print("\n최적 정책 (4x4 그리드):")
    print("S: 시작, H: 구멍, F: 얼음, G: 목표")
    for i in range(4):
        row = ""
        for j in range(4):
            s = i * 4 + j
            if s == 0:
                row += "  S  "
            elif s in [5, 7, 11, 12]:  # 구멍
                row += "  H  "
            elif s == 15:  # 목표
                row += "  G  "
            else:
                row += f"  {action_names[policy[s]]}  "
        print(row)

    print("\n가치 함수:")
    print(V.reshape(4, 4).round(3))

    # 정책 테스트
    print("\n정책 테스트 중...")
    success = 0
    n_tests = 100

    for _ in range(n_tests):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward > 0:
            success += 1

    print(f"성공률: {success}/{n_tests} = {success/n_tests*100:.1f}%")

    env.close()
    return V, policy


if __name__ == "__main__":
    # 그리드 월드 알고리즘 비교
    V_pi, policy_pi, V_vi, policy_vi = compare_algorithms()

    # Frozen Lake 예제
    try:
        V_fl, policy_fl = frozen_lake_example()
    except Exception as e:
        print(f"\nFrozen Lake 예제 실행 실패: {e}")
        print("gymnasium 패키지가 설치되어 있는지 확인하세요: pip install gymnasium")

    print("\n" + "=" * 60)
    print("동적 프로그래밍 예제 완료!")
    print("=" * 60)
