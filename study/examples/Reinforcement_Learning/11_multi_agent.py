"""
Multi-Agent RL: IQL과 간단한 협력/경쟁 환경
다중 에이전트 강화학습의 기본 개념 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class IQLAgent:
    """Independent Q-Learning 에이전트"""

    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def choose_action(self, obs):
        """Epsilon-greedy 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(obs))
            return q_values.argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        """Q-learning 업데이트"""
        obs_tensor = torch.FloatTensor(obs)
        next_obs_tensor = torch.FloatTensor(next_obs)

        current_q = self.q_network(obs_tensor)[action]

        with torch.no_grad():
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * self.q_network(next_obs_tensor).max()

        loss = (current_q - target_q) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class SimpleGridWorld:
    """
    간단한 2-에이전트 그리드 환경
    - 그리드: 5x5
    - 목표: 두 에이전트가 각자의 목표 지점에 도달
    - 협력 요소: 같은 셀에 있으면 보너스 보상
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_agents = 2

        # 행동: 상, 하, 좌, 우, 대기
        self.action_dim = 5
        self.obs_dim = 4  # (x, y, goal_x, goal_y)

        self.reset()

    def reset(self):
        """환경 초기화"""
        # 에이전트 초기 위치 (무작위)
        self.agent_pos = [
            [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            for _ in range(self.n_agents)
        ]

        # 목표 위치 (고정)
        self.goals = [
            [0, self.grid_size - 1],  # 에이전트 0의 목표
            [self.grid_size - 1, 0]   # 에이전트 1의 목표
        ]

        self.steps = 0
        return self.get_observations()

    def get_observations(self):
        """각 에이전트의 관측 반환"""
        observations = []
        for i in range(self.n_agents):
            obs = [
                self.agent_pos[i][0] / self.grid_size,
                self.agent_pos[i][1] / self.grid_size,
                self.goals[i][0] / self.grid_size,
                self.goals[i][1] / self.grid_size
            ]
            observations.append(np.array(obs, dtype=np.float32))
        return observations

    def step(self, actions):
        """환경 스텝"""
        self.steps += 1
        rewards = [0.0, 0.0]

        # 각 에이전트 이동
        for i, action in enumerate(actions):
            x, y = self.agent_pos[i]

            # 행동 적용: 상(0), 하(1), 좌(2), 우(3), 대기(4)
            if action == 0:  # 상
                x = max(0, x - 1)
            elif action == 1:  # 하
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # 좌
                y = max(0, y - 1)
            elif action == 3:  # 우
                y = min(self.grid_size - 1, y + 1)
            # action == 4: 대기

            self.agent_pos[i] = [x, y]

            # 목표 도달 보상
            if self.agent_pos[i] == self.goals[i]:
                rewards[i] += 10.0

            # 매 스텝마다 작은 페널티
            rewards[i] -= 0.01

        # 협력 보너스: 같은 셀에 있으면
        if self.agent_pos[0] == self.agent_pos[1]:
            rewards[0] += 1.0
            rewards[1] += 1.0

        # 종료 조건: 둘 다 목표 도달 또는 최대 스텝 도달
        done = (
            (self.agent_pos[0] == self.goals[0] and self.agent_pos[1] == self.goals[1])
            or self.steps >= 50
        )

        observations = self.get_observations()
        return observations, rewards, done


class CompetitiveGridWorld:
    """
    경쟁 환경: 두 에이전트가 하나의 보상을 두고 경쟁
    먼저 도달한 에이전트가 보상을 가져감
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_agents = 2
        self.action_dim = 5
        self.obs_dim = 4

        self.reset()

    def reset(self):
        """환경 초기화"""
        # 에이전트 초기 위치
        self.agent_pos = [
            [0, 0],
            [self.grid_size - 1, self.grid_size - 1]
        ]

        # 공통 목표 위치 (중앙)
        self.goal = [self.grid_size // 2, self.grid_size // 2]
        self.goal_taken = False
        self.steps = 0

        return self.get_observations()

    def get_observations(self):
        """각 에이전트의 관측 반환"""
        observations = []
        for i in range(self.n_agents):
            obs = [
                self.agent_pos[i][0] / self.grid_size,
                self.agent_pos[i][1] / self.grid_size,
                self.goal[0] / self.grid_size,
                self.goal[1] / self.grid_size
            ]
            observations.append(np.array(obs, dtype=np.float32))
        return observations

    def step(self, actions):
        """환경 스텝"""
        self.steps += 1
        rewards = [0.0, 0.0]

        # 각 에이전트 이동
        for i, action in enumerate(actions):
            x, y = self.agent_pos[i]

            if action == 0:  # 상
                x = max(0, x - 1)
            elif action == 1:  # 하
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # 좌
                y = max(0, y - 1)
            elif action == 3:  # 우
                y = min(self.grid_size - 1, y + 1)

            self.agent_pos[i] = [x, y]

            # 목표 도달 체크 (먼저 도달한 에이전트만 보상)
            if not self.goal_taken and self.agent_pos[i] == self.goal:
                rewards[i] += 10.0
                self.goal_taken = True

            # 매 스텝마다 작은 페널티
            rewards[i] -= 0.01

        # 종료 조건
        done = self.goal_taken or self.steps >= 50

        observations = self.get_observations()
        return observations, rewards, done


class IQLSystem:
    """다중 에이전트 IQL 시스템"""

    def __init__(self, n_agents, obs_dim, action_dim):
        self.agents = [
            IQLAgent(obs_dim, action_dim)
            for _ in range(n_agents)
        ]
        self.n_agents = n_agents

    def choose_actions(self, observations):
        """모든 에이전트의 행동 선택"""
        return [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update(self, observations, actions, rewards, next_observations, done):
        """모든 에이전트 업데이트"""
        losses = []
        for i, agent in enumerate(self.agents):
            loss = agent.update(
                observations[i], actions[i],
                rewards[i], next_observations[i], done
            )
            losses.append(loss)
        return losses


def train_cooperative():
    """협력 환경에서 IQL 학습"""
    print("=== 협력 환경 학습 ===\n")

    env = SimpleGridWorld(grid_size=5)
    system = IQLSystem(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )

    n_episodes = 1000
    episode_rewards = []

    for episode in range(n_episodes):
        observations = env.reset()
        total_rewards = [0.0, 0.0]
        done = False

        while not done:
            actions = system.choose_actions(observations)
            next_observations, rewards, done = env.step(actions)

            system.update(observations, actions, rewards, next_observations, done)

            observations = next_observations
            total_rewards[0] += rewards[0]
            total_rewards[1] += rewards[1]

        episode_rewards.append(sum(total_rewards) / 2)

        # Epsilon 감소
        for agent in system.agents:
            agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards


def train_competitive():
    """경쟁 환경에서 IQL 학습"""
    print("\n=== 경쟁 환경 학습 ===\n")

    env = CompetitiveGridWorld(grid_size=5)
    system = IQLSystem(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )

    n_episodes = 1000
    agent0_wins = []
    agent1_wins = []

    for episode in range(n_episodes):
        observations = env.reset()
        episode_rewards = [0.0, 0.0]
        done = False

        while not done:
            actions = system.choose_actions(observations)
            next_observations, rewards, done = env.step(actions)

            system.update(observations, actions, rewards, next_observations, done)

            observations = next_observations
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]

        # 승자 기록
        agent0_wins.append(1 if episode_rewards[0] > episode_rewards[1] else 0)
        agent1_wins.append(1 if episode_rewards[1] > episode_rewards[0] else 0)

        # Epsilon 감소
        for agent in system.agents:
            agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if (episode + 1) % 100 == 0:
            win_rate_0 = np.mean(agent0_wins[-100:]) * 100
            win_rate_1 = np.mean(agent1_wins[-100:]) * 100
            print(f"Episode {episode + 1}, Win Rate - Agent0: {win_rate_0:.1f}%, Agent1: {win_rate_1:.1f}%")

    return agent0_wins, agent1_wins


def visualize_results(coop_rewards, comp_wins):
    """학습 결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 협력 환경 보상
    axes[0].plot(coop_rewards, alpha=0.3, color='blue')
    window = 50
    smoothed = np.convolve(coop_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, color='blue', linewidth=2)
    axes[0].set_title('협력 환경: 평균 보상')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].grid(True, alpha=0.3)

    # 경쟁 환경 승률
    agent0_wins, agent1_wins = comp_wins
    window = 50
    win_rate_0 = np.convolve(agent0_wins, np.ones(window)/window, mode='valid') * 100
    win_rate_1 = np.convolve(agent1_wins, np.ones(window)/window, mode='valid') * 100

    axes[1].plot(win_rate_0, label='Agent 0', linewidth=2)
    axes[1].plot(win_rate_1, label='Agent 1', linewidth=2)
    axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.3, label='균형점')
    axes[1].set_title('경쟁 환경: 승률')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multi_agent_results.png', dpi=100, bbox_inches='tight')
    print("\n그래프 저장: multi_agent_results.png")


def demonstrate_ctde_concept():
    """
    CTDE (Centralized Training, Decentralized Execution) 개념 설명
    훈련 시에는 글로벌 정보를 사용하지만, 실행 시에는 로컬 관측만 사용
    """
    print("\n=== CTDE 패러다임 개념 ===\n")
    print("훈련 단계:")
    print("  - Critic: 모든 에이전트의 관측 + 행동에 접근 가능")
    print("  - 글로벌 상태로 가치 함수 학습")
    print("\n실행 단계:")
    print("  - Actor: 로컬 관측만 사용")
    print("  - 분산 실행으로 통신 불필요")
    print("\n장점:")
    print("  - 학습 시 협력 패턴 발견 용이")
    print("  - 실행 시 확장성 좋음")
    print("  - 부분 관측 환경에서도 작동")


if __name__ == "__main__":
    print("다중 에이전트 강화학습 예제\n")

    # 협력 환경 학습
    coop_rewards = train_cooperative()

    # 경쟁 환경 학습
    agent0_wins, agent1_wins = train_competitive()

    # CTDE 개념 설명
    demonstrate_ctde_concept()

    # 결과 시각화
    visualize_results(coop_rewards, (agent0_wins, agent1_wins))

    print("\n학습 완료!")
    print("\n주요 개념:")
    print("1. IQL: 각 에이전트가 독립적으로 Q-learning")
    print("2. 비정상성: 다른 에이전트의 정책 변화로 환경이 동적")
    print("3. 협력 vs 경쟁: 보상 구조에 따른 학습 양상 차이")
    print("4. CTDE: 중앙집중 학습, 분산 실행 패러다임")
