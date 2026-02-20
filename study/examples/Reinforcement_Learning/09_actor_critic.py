"""
Actor-Critic (A2C) 구현
Actor-Critic 아키텍처, Advantage 추정, GAE 포함
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
    """Actor-Critic 공유 네트워크"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 공유 특징 추출 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (정책 네트워크)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (가치 네트워크)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """순전파: 정책과 가치를 동시에 출력"""
        features = self.shared(state)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

    def get_action(self, state):
        """행동 샘플링"""
        policy_logits, value = self.forward(state)
        policy = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value, entropy


class A2CAgent:
    """A2C (Advantage Actor-Critic) 에이전트"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01):
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.value_coef = value_coef  # Critic 손실 가중치
        self.entropy_coef = entropy_coef  # 엔트로피 보너스 가중치

        # 에피소드 버퍼
        self.reset_buffers()

    def reset_buffers(self):
        """버퍼 초기화"""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def choose_action(self, state):
        """행동 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value, entropy = self.network.get_action(state_tensor)

        # 버퍼에 저장
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

        return action

    def store_transition(self, reward, done):
        """전이 저장"""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, next_value):
        """n-step returns 계산 (부트스트래핑)"""
        returns = []
        R = next_value

        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32)

    def update(self, next_state):
        """A2C 업데이트"""
        if len(self.rewards) == 0:
            return 0, 0

        # 다음 상태의 가치 (부트스트래핑용)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.network(state_tensor)
            next_value = next_value.item()

        # Returns 계산
        returns = self.compute_returns(next_value)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Advantage 계산: A(s,a) = Q(s,a) - V(s) ≈ R - V(s)
        advantages = returns - values.detach()

        # 손실 계산
        actor_loss = -(log_probs * advantages).mean()  # Policy gradient
        critic_loss = F.mse_loss(values, returns)  # Value function loss
        entropy_loss = -entropies.mean()  # 탐색 장려

        total_loss = (actor_loss +
                      self.value_coef * critic_loss +
                      self.entropy_coef * entropy_loss)

        # 그래디언트 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        # 버퍼 초기화
        self.reset_buffers()

        return actor_loss.item(), critic_loss.item()


class A2CWithGAE(A2CAgent):
    """GAE (Generalized Advantage Estimation)를 사용하는 A2C"""

    def __init__(self, *args, gae_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda

    def compute_gae(self, next_value):
        """GAE를 사용한 Advantage 계산"""
        values = torch.cat(self.values).squeeze().tolist()
        values.append(next_value)  # 마지막에 부트스트랩 가치 추가

        advantages = []
        gae = 0

        # 역방향으로 GAE 계산
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

        return advantages, returns

    def update(self, next_state):
        """GAE를 사용한 업데이트"""
        if len(self.rewards) == 0:
            return 0, 0

        # 다음 상태의 가치
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.network(state_tensor)
            next_value = next_value.item()

        # GAE로 advantage와 returns 계산
        advantages, returns = self.compute_gae(next_value)

        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # 손실 계산
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        total_loss = (actor_loss +
                      self.value_coef * critic_loss +
                      self.entropy_coef * entropy_loss)

        # 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.reset_buffers()

        return actor_loss.item(), critic_loss.item()


def train_a2c(env_name='CartPole-v1', n_episodes=1000, n_steps=5, use_gae=False):
    """A2C 학습"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 에이전트 생성
    if use_gae:
        agent = A2CWithGAE(state_dim, action_dim, lr=3e-4, gamma=0.99,
                          value_coef=0.5, entropy_coef=0.01, gae_lambda=0.95)
        method_name = "A2C with GAE"
    else:
        agent = A2CAgent(state_dim, action_dim, lr=3e-4, gamma=0.99,
                        value_coef=0.5, entropy_coef=0.01)
        method_name = "A2C"

    print(f"=== {method_name} 학습 시작 ({env_name}) ===\n")

    scores = []
    actor_losses = []
    critic_losses = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(reward, done)
            state = next_state
            total_reward += reward
            step_count += 1

            # n-step 업데이트 또는 에피소드 종료 시 업데이트
            if step_count % n_steps == 0 or done:
                actor_loss, critic_loss = agent.update(next_state)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        scores.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_actor_loss = np.mean(actor_losses[-50:]) if actor_losses else 0
            avg_critic_loss = np.mean(critic_losses[-50:]) if critic_losses else 0
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Score: {avg_score:7.2f} | "
                  f"Actor Loss: {avg_actor_loss:.4f} | "
                  f"Critic Loss: {avg_critic_loss:.4f}")

        # CartPole 해결 조건: 연속 100 에피소드 평균 475 이상
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 475:
            print(f"\n환경 해결! ({episode + 1} 에피소드)")
            break

    env.close()
    return agent, scores, actor_losses, critic_losses


def compare_a2c_with_reinforce():
    """A2C와 REINFORCE 비교"""
    print("=== A2C vs REINFORCE 비교 ===\n")

    # A2C 학습
    _, a2c_scores, _, _ = train_a2c('CartPole-v1', n_episodes=500, use_gae=False)

    # A2C with GAE 학습
    print("\n" + "="*60 + "\n")
    _, a2c_gae_scores, _, _ = train_a2c('CartPole-v1', n_episodes=500, use_gae=True)

    # 학습 곡선 비교 시각화
    plot_comparison(a2c_scores, a2c_gae_scores)

    return a2c_scores, a2c_gae_scores


def plot_comparison(a2c_scores, a2c_gae_scores):
    """학습 곡선 비교 시각화"""
    window = 10

    def smooth(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(14, 5))

    # 원본 데이터
    plt.subplot(1, 2, 1)
    plt.plot(a2c_scores, alpha=0.3, label='A2C (raw)', color='blue')
    plt.plot(a2c_gae_scores, alpha=0.3, label='A2C+GAE (raw)', color='green')
    plt.axhline(y=475, color='red', linestyle='--', linewidth=1, label='Solved threshold')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('A2C vs A2C+GAE - Raw Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 평활화된 데이터
    plt.subplot(1, 2, 2)
    plt.plot(smooth(a2c_scores, window), label='A2C (smoothed)', linewidth=2, color='blue')
    plt.plot(smooth(a2c_gae_scores, window), label='A2C+GAE (smoothed)', linewidth=2, color='green')
    plt.axhline(y=475, color='red', linestyle='--', linewidth=1, label='Solved threshold')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (smoothed)')
    plt.title(f'A2C vs A2C+GAE - Smoothed (window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('a2c_comparison.png', dpi=150)
    print("\n학습 곡선이 'a2c_comparison.png'로 저장되었습니다.")


def train_lunarlander():
    """LunarLander 환경에서 A2C 학습"""
    try:
        env = gym.make('LunarLander-v2')
    except:
        print("LunarLander-v2 환경을 찾을 수 없습니다.")
        print("설치: pip install gymnasium[box2d]")
        return None, None

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # GAE를 사용하는 A2C
    agent = A2CWithGAE(
        state_dim, action_dim,
        lr=7e-4, gamma=0.99,
        value_coef=0.5, entropy_coef=0.01,
        gae_lambda=0.95
    )

    print("=== LunarLander A2C 학습 시작 ===\n")

    scores = []
    n_steps = 5
    n_episodes = 2000

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(reward, done)
            state = next_state
            total_reward += reward
            steps += 1

            # n-step 업데이트 또는 종료 시
            if steps % n_steps == 0 or done:
                agent.update(next_state)

            if done:
                break

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {episode + 1:4d} | Avg Score: {avg:.2f}")

            # LunarLander 해결 조건: 평균 200 이상
            if avg >= 200:
                print(f"\n환경 해결! ({episode + 1} 에피소드)")
                break

    env.close()
    return agent, scores


if __name__ == "__main__":
    # 1. CartPole에서 A2C vs A2C+GAE 비교
    a2c_scores, a2c_gae_scores = compare_a2c_with_reinforce()

    # 2. LunarLander 학습 (선택적)
    print("\n" + "="*60)
    print("LunarLander 학습을 시작하려면 주석을 해제하세요:")
    print("# agent, scores = train_lunarlander()")

    # 학습 결과 요약
    print("\n" + "="*60)
    print("학습 완료!")
    print(f"A2C 최종 100 에피소드 평균: {np.mean(a2c_scores[-100:]):.2f}")
    print(f"A2C+GAE 최종 100 에피소드 평균: {np.mean(a2c_gae_scores[-100:]):.2f}")
