"""
PPO (Proximal Policy Optimization) 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        return F.softmax(self.actor(features), dim=-1), self.critic(features)

    def get_action_and_value(self, state, action=None):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, epochs=10, batch_size=64):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def collect_rollout(self, env, n_steps):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        obs, _ = env.reset()

        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, logp, _, value = self.network.get_action_and_value(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action.item())
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value.item())
            logp_buf.append(logp.item())

            obs = next_obs if not done else env.reset()[0]

        with torch.no_grad():
            _, _, _, last_value = self.network.get_action_and_value(
                torch.FloatTensor(obs).unsqueeze(0)
            )

        return {
            'obs': np.array(obs_buf), 'actions': np.array(act_buf),
            'rewards': np.array(rew_buf), 'dones': np.array(done_buf),
            'values': np.array(val_buf), 'log_probs': np.array(logp_buf),
            'last_value': last_value.item()
        }

    def compute_gae(self, rollout):
        rewards, values, dones = rollout['rewards'], rollout['values'], rollout['dones']
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_val = rollout['last_value'] if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        return advantages, advantages + values

    def update(self, rollout):
        advantages, returns = self.compute_gae(rollout)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.FloatTensor(rollout['obs'])
        actions = torch.LongTensor(rollout['actions'])
        old_logp = torch.FloatTensor(rollout['log_probs'])
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        for _ in range(self.epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                idx = indices[start:start + self.batch_size]

                _, new_logp, entropy, values = self.network.get_action_and_value(
                    obs[idx], actions[idx]
                )

                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, returns[idx])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()


def train():
    env = gym.make('CartPole-v1')
    agent = PPO(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    n_steps = 128
    timesteps = 0
    episode_rewards = []
    current_reward = 0

    while timesteps < 50000:
        rollout = agent.collect_rollout(env, n_steps)
        timesteps += n_steps

        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_reward += r
            if d:
                episode_rewards.append(current_reward)
                current_reward = 0

        agent.update(rollout)

        if len(episode_rewards) > 0 and timesteps % 5000 < n_steps:
            print(f"Timesteps: {timesteps}, Avg: {np.mean(episode_rewards[-10:]):.2f}")

    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    agent, rewards = train()
