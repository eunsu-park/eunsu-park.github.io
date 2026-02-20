"""
Soft Actor-Critic (SAC) for Continuous Control
===============================================

SAC is a state-of-the-art off-policy RL algorithm for continuous action spaces.
Key features:
1. Maximum Entropy RL: encourages exploration through entropy regularization
2. Twin Q-networks: reduces overestimation bias
3. Squashed Gaussian policy: bounded continuous actions
4. Automatic temperature tuning: adaptive entropy coefficient

This simplified implementation demonstrates core SAC concepts on Pendulum-v1.

Requirements: torch, gymnasium, numpy, matplotlib
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Tuple, List


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store transition in buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch from buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of batched (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class GaussianActor(nn.Module):
    """
    Gaussian policy network with tanh squashing.
    Outputs mean and log_std for a Gaussian distribution.
    Actions are sampled and squashed to [-1, 1] range.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize actor network.

        Args:
            state_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            state: State tensor

        Returns:
            (mean, log_std) for Gaussian distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and compute log probability.

        Args:
            state: State tensor

        Returns:
            (action, log_prob) where action is squashed to [-1, 1]
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick

        # Squash to [-1, 1] using tanh
        action = torch.tanh(x)

        # Compute log probability with change of variables correction
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  # tanh correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class TwinQCritic(nn.Module):
    """
    Twin Q-networks to reduce overestimation bias.
    Both networks have identical architecture but independent parameters.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize twin Q-networks.

        Args:
            state_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
        """
        super().__init__()

        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both networks.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            (Q1_value, Q2_value)
        """
        x = torch.cat([state, action], dim=-1)

        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2


class SACAgent:
    """
    Soft Actor-Critic agent with automatic temperature tuning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: Observation dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Polyak averaging coefficient for target networks
            alpha: Entropy coefficient (initial value if auto-tuning)
            auto_tune_alpha: Whether to automatically tune alpha
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.auto_tune_alpha = auto_tune_alpha

        # Actor network
        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Q-networks
        self.critic = TwinQCritic(state_dim, action_dim).to(self.device)
        self.critic_target = TwinQCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic temperature tuning
        if self.auto_tune_alpha:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action from policy.

        Args:
            state: Current state
            deterministic: If True, return mean action (for evaluation)

        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def update(self, batch_size: int, replay_buffer: ReplayBuffer):
        """
        Update networks using a batch from replay buffer.

        Args:
            batch_size: Batch size for sampling
            replay_buffer: Replay buffer to sample from
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ==================== Update Critic ====================
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values using target networks
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)

            # Add entropy term
            target_q = rewards + (1 - dones) * self.gamma * (min_q_target - self.alpha * next_log_probs)

        # Current Q-values
        q1, q2 = self.critic(states, actions)

        # Critic loss: MSE between current and target Q-values
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ==================== Update Actor ====================
        # Sample actions from current policy
        new_actions, log_probs = self.actor.sample(states)

        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Actor loss: maximize Q - alpha * log_prob (equivalent to minimize negative)
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ==================== Update Temperature ====================
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ==================== Update Target Networks ====================
        # Polyak averaging: target = tau * current + (1 - tau) * target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_sac(
    env_name: str = 'Pendulum-v1',
    max_episodes: int = 100,
    max_steps: int = 200,
    batch_size: int = 128,
    buffer_capacity: int = 100000,
    warmup_steps: int = 1000
) -> List[float]:
    """
    Train SAC agent on continuous control environment.

    Args:
        env_name: Gymnasium environment name
        max_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size for updates
        buffer_capacity: Replay buffer capacity
        warmup_steps: Random exploration steps before training

    Returns:
        List of episode rewards
    """
    # Create environment
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent and replay buffer
    agent = SACAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []
    total_steps = 0

    print(f"Training SAC on {env_name}...")
    print(f"State dim: {state_dim}, Action dim: {action_dim}\n")

    # Warmup: collect random transitions
    state, _ = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    print(f"Warmup complete: {warmup_steps} steps collected\n")

    # Training loop
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)

            # Update agent
            agent.update(batch_size, replay_buffer)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{max_episodes}, "
                  f"Avg Reward (last 10): {avg_reward:.2f}, "
                  f"Alpha: {agent.alpha:.3f}")

    env.close()
    return episode_rewards


def plot_results(rewards: List[float]):
    """
    Plot learning curve.

    Args:
        rewards: List of episode rewards
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')

    # Moving average
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, linewidth=2, label=f'{window}-Episode Moving Average')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('SAC Training on Pendulum-v1', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Reinforcement_Learning/sac_training.png', dpi=150)
    print(f"\nPlot saved to: sac_training.png")
    plt.show()


if __name__ == '__main__':
    # Train agent
    rewards = train_sac(
        env_name='Pendulum-v1',
        max_episodes=100,
        max_steps=200,
        batch_size=128,
        warmup_steps=1000
    )

    # Plot results
    plot_results(rewards)

    # Print final performance
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final 10-episode average reward: {np.mean(rewards[-10:]):.2f}")
    print(f"Best 10-episode average reward: {max([np.mean(rewards[i:i+10]) for i in range(len(rewards)-10)]):.2f}")
