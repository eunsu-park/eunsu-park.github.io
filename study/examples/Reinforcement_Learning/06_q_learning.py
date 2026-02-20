"""
Q-Learning과 SARSA 구현
"""
import numpy as np
import gymnasium as gym


class QLearning:
    """Q-Learning 에이전트"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


class SARSA:
    """SARSA 에이전트"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


def train_qlearning():
    """Q-Learning으로 FrozenLake 학습"""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )

    n_episodes = 10000
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        agent.epsilon = max(0.01, agent.epsilon * 0.9995)

        if (episode + 1) % 1000 == 0:
            avg = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg:.3f}")

    env.close()
    return agent


if __name__ == "__main__":
    agent = train_qlearning()
    print("\n학습 완료!")
    print(f"최종 Q 테이블 shape: {agent.q_table.shape}")
