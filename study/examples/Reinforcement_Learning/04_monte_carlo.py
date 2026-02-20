"""
몬테카를로 방법 (Monte Carlo Methods) 구현
- First-visit MC Prediction
- Every-visit MC Prediction
- MC Control (Exploring Starts)
- On-policy MC Control (ε-greedy)
- Off-policy MC (Importance Sampling)
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym


def calculate_returns(episode, gamma=0.99):
    """
    에피소드에서 각 시점의 리턴 계산

    Args:
        episode: [(state, action, reward), ...] 형태의 리스트
        gamma: 할인율

    Returns:
        returns: [(state, G), ...] 각 시점의 리턴
    """
    G = 0  # 리턴 초기화
    returns = []

    # 역순으로 계산 (효율적인 계산)
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = reward + gamma * G  # 할인된 리턴
        returns.insert(0, (state, G))

    return returns


def first_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    First-visit MC 정책 평가

    Args:
        env: Gymnasium 환경
        policy: 정책 함수 policy(state) -> action
        n_episodes: 에피소드 수
        gamma: 할인율

    Returns:
        V: 상태 가치 함수
    """
    # 각 상태의 리턴 합과 방문 횟수
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        # 에피소드 생성
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # First-visit: 각 상태의 첫 방문 인덱스 찾기
        visited = set()
        G = 0

        # 역순으로 리턴 계산
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # First-visit 체크
            if state_t not in visited:
                visited.add(state_t)
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]

        if (episode_num + 1) % 2000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return dict(V)


def every_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    Every-visit MC 정책 평가

    모든 방문을 카운트
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        G = 0

        # Every-visit: 모든 방문에서 업데이트
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # 모든 방문 카운트
            returns_sum[state_t] += G
            returns_count[state_t] += 1
            V[state_t] = returns_sum[state_t] / returns_count[state_t]

        if (episode_num + 1) % 2000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return dict(V)


def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    """
    ε-탐욕 행동 선택

    Args:
        Q: 행동 가치 함수
        state: 현재 상태
        n_actions: 행동 수
        epsilon: 탐험 확률

    Returns:
        action: 선택된 행동
    """
    if np.random.random() < epsilon:
        # 탐험: 랜덤 행동
        return np.random.randint(n_actions)
    else:
        # 활용: 최선의 행동
        return np.argmax(Q[state])


def mc_on_policy_control(env, n_episodes=100000, gamma=0.99,
                         epsilon=0.1, epsilon_decay=0.9999):
    """
    On-policy MC 제어 (ε-greedy)

    Args:
        env: Gymnasium 환경
        n_episodes: 에피소드 수
        gamma: 할인율
        epsilon: 탐험율
        epsilon_decay: epsilon 감소율

    Returns:
        Q: 행동 가치 함수
        policy: 학습된 정책
        episode_rewards: 에피소드별 보상
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    episode_rewards = []

    print("MC On-Policy Control 학습 시작...")
    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0

        # ε-greedy 정책으로 에피소드 생성
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, action, reward))
            total_reward += reward

            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        # Q 업데이트 (First-visit)
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        # epsilon 감소
        epsilon = max(0.01, epsilon * epsilon_decay)

        if (episode_num + 1) % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode_num + 1}: avg_reward = {avg_reward:.3f}, "
                  f"epsilon = {epsilon:.4f}")

    # 최종 탐욕적 정책
    policy = {}
    for state in Q:
        policy[state] = np.argmax(Q[state])

    return dict(Q), policy, episode_rewards


def mc_off_policy_control(env, n_episodes=100000, gamma=0.99):
    """
    Off-policy MC 제어 (Weighted Importance Sampling)

    행동 정책: ε-greedy (탐험)
    목표 정책: greedy (활용)

    Returns:
        Q: 행동 가치 함수
        target_policy: 목표 정책
        episode_rewards: 에피소드별 보상
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    C = defaultdict(lambda: np.zeros(n_actions))  # 가중치 합

    episode_rewards = []
    epsilon = 0.1  # 행동 정책의 epsilon

    print("MC Off-Policy Control 학습 시작...")
    for episode_num in range(n_episodes):
        # 행동 정책 (ε-greedy)으로 에피소드 생성
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        G = 0
        W = 1.0  # 중요도 샘플링 가중치

        # 역순 처리
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            # 가중 중요도 샘플링 업데이트
            C[state_t][action_t] += W
            Q[state_t][action_t] += (W / C[state_t][action_t] *
                                     (G - Q[state_t][action_t]))

            # 목표 정책에서의 행동 (greedy)
            target_action = np.argmax(Q[state_t])

            # 행동이 목표 정책과 다르면 중단
            if action_t != target_action:
                break

            # 중요도 비율 업데이트
            # π(a|s) = 1 (결정적), b(a|s) = (1-ε) + ε/|A| or ε/|A|
            if action_t == target_action:
                b_prob = (1 - epsilon) + epsilon / n_actions
            else:
                b_prob = epsilon / n_actions

            W = W * 1.0 / b_prob

        if (episode_num + 1) % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode_num + 1}: avg_reward = {avg_reward:.3f}")

    # 최종 탐욕적 정책
    target_policy = {}
    for state in Q:
        target_policy[state] = np.argmax(Q[state])

    return dict(Q), target_policy, episode_rewards


def blackjack_example():
    """블랙잭 환경에서 MC 학습"""
    print("\n" + "=" * 60)
    print("블랙잭 예제 - MC On-Policy Control")
    print("=" * 60)

    env = gym.make('Blackjack-v1', sab=True)

    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    n_episodes = 500000
    gamma = 1.0
    epsilon = 0.1

    wins = 0
    losses = 0
    draws = 0

    print(f"\n{n_episodes} 에피소드 학습 중...")
    for ep in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        # 에피소드 생성
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # 결과 기록
        final_reward = episode[-1][2]
        if final_reward == 1:
            wins += 1
        elif final_reward == -1:
            losses += 1
        else:
            draws += 1

        # Q 업데이트
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        if (ep + 1) % 100000 == 0:
            win_rate = wins / (ep + 1)
            print(f"Episode {ep + 1}: 승률 = {win_rate:.3f}")

    env.close()

    # 최종 통계
    print("\n학습 완료!")
    print(f"총 에피소드: {n_episodes}")
    print(f"승리: {wins} ({wins/n_episodes*100:.1f}%)")
    print(f"패배: {losses} ({losses/n_episodes*100:.1f}%)")
    print(f"무승부: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"학습된 상태-행동 쌍 수: {len(Q)}")

    # 정책 시각화
    visualize_blackjack_policy(Q)

    return Q


def visualize_blackjack_policy(Q):
    """블랙잭 정책 시각화"""
    print("\n" + "=" * 60)
    print("학습된 블랙잭 정책")
    print("=" * 60)
    print("H: Hit (카드 추가), S: Stick (패 유지)")

    print("\n=== 사용 가능한 에이스가 없을 때 ===")
    print("       딜러 카드")
    print("합계   A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 11, -1):
        row = f"{player_sum:2d}:   "
        for dealer in range(1, 11):
            state = (player_sum, dealer, False)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)

    print("\n=== 사용 가능한 에이스가 있을 때 ===")
    print("       딜러 카드")
    print("합계   A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 11, -1):
        row = f"{player_sum:2d}:   "
        for dealer in range(1, 11):
            state = (player_sum, dealer, True)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)


def plot_learning_curve(episode_rewards, window=1000):
    """학습 곡선 시각화"""
    # 이동 평균 계산
    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(episode_rewards[start:i+1]))

    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg, label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Monte Carlo Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mc_learning_curve.png', dpi=150)
    print("학습 곡선 저장: mc_learning_curve.png")


def compare_mc_methods():
    """MC 방법 비교"""
    print("=" * 60)
    print("몬테카를로 방법 비교")
    print("=" * 60)

    env = gym.make('Blackjack-v1', sab=True)

    # 1. First-visit MC Prediction
    print("\n[1] First-visit MC Prediction (랜덤 정책)")
    print("-" * 60)

    def random_policy(state):
        return env.action_space.sample()

    V_first = first_visit_mc_prediction(env, random_policy, n_episodes=10000)
    print(f"추정된 상태 수: {len(V_first)}")
    print(f"샘플 상태 가치: {list(V_first.items())[:5]}")

    # 2. Every-visit MC Prediction
    print("\n[2] Every-visit MC Prediction (랜덤 정책)")
    print("-" * 60)
    V_every = every_visit_mc_prediction(env, random_policy, n_episodes=10000)
    print(f"추정된 상태 수: {len(V_every)}")

    # 3. On-policy MC Control
    print("\n[3] On-policy MC Control (ε-greedy)")
    print("-" * 60)
    Q_on, policy_on, rewards_on = mc_on_policy_control(
        env, n_episodes=50000, gamma=1.0, epsilon=0.1
    )
    print(f"학습된 상태-행동 쌍 수: {len(Q_on)}")
    print(f"최종 평균 보상: {np.mean(rewards_on[-1000:]):.3f}")

    # 4. Off-policy MC Control
    print("\n[4] Off-policy MC Control (Importance Sampling)")
    print("-" * 60)
    Q_off, policy_off, rewards_off = mc_off_policy_control(
        env, n_episodes=50000, gamma=1.0
    )
    print(f"학습된 상태-행동 쌍 수: {len(Q_off)}")
    print(f"최종 평균 보상: {np.mean(rewards_off[-1000:]):.3f}")

    env.close()

    # 학습 곡선 비교
    plt.figure(figsize=(12, 6))

    window = 1000
    moving_avg_on = [np.mean(rewards_on[max(0, i-window+1):i+1])
                     for i in range(len(rewards_on))]
    moving_avg_off = [np.mean(rewards_off[max(0, i-window+1):i+1])
                      for i in range(len(rewards_off))]

    plt.plot(moving_avg_on, label='On-policy MC', alpha=0.7)
    plt.plot(moving_avg_off, label='Off-policy MC', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('On-policy vs Off-policy MC Control')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mc_comparison.png', dpi=150)
    print("\n비교 그래프 저장: mc_comparison.png")

    return Q_on, policy_on, Q_off, policy_off


if __name__ == "__main__":
    # MC 방법 비교
    try:
        Q_on, policy_on, Q_off, policy_off = compare_mc_methods()

        # 블랙잭 예제
        Q_blackjack = blackjack_example()

    except Exception as e:
        print(f"\n실행 실패: {e}")
        print("gymnasium 패키지가 설치되어 있는지 확인하세요: pip install gymnasium")

    print("\n" + "=" * 60)
    print("몬테카를로 방법 예제 완료!")
    print("=" * 60)
