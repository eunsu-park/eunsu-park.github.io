"""
Dyna-Q: Model-Based Reinforcement Learning
===========================================

Dyna-Q combines model-free Q-learning with model-based planning.
The agent learns:
1. Q-values from real experience (Q-learning)
2. A model of the environment (transition dynamics)
3. Uses the model to generate simulated experience for additional learning

This demonstrates the sample efficiency benefit of model-based planning.

Requirements: gymnasium, numpy, matplotlib
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, List


class DynaQ:
    """
    Dyna-Q agent with tabular Q-learning and model-based planning.

    The agent maintains:
    - Q-table: state-action values
    - Model: transition dynamics (s, a) -> (s', r)
    - Memory: visited state-action pairs for planning
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        n_planning: int = 5
    ):
        """
        Initialize Dyna-Q agent.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Q-learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            n_planning: Number of planning steps per real step
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning

        # Q-table: Q[s, a]
        self.Q = np.zeros((n_states, n_actions))

        # Model: stores (next_state, reward) for each (state, action)
        # Using defaultdict to handle unseen state-action pairs
        self.model = {}

        # Memory: set of visited (state, action) pairs
        self.memory = set()

    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update_q(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-value using Q-learning rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Q-learning update
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def update_model(self, state: int, action: int, reward: float, next_state: int):
        """
        Update model with new transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Store transition in model
        self.model[(state, action)] = (next_state, reward)
        # Add to memory for planning
        self.memory.add((state, action))

    def plan(self):
        """
        Perform model-based planning by sampling from model.

        For n_planning steps:
        1. Sample a previously visited (s, a) pair
        2. Use model to get (s', r)
        3. Update Q-value with simulated experience
        """
        for _ in range(self.n_planning):
            if not self.memory:
                break

            # Sample random state-action from memory
            state, action = list(self.memory)[np.random.randint(len(self.memory))]

            # Get predicted next state and reward from model
            next_state, reward = self.model[(state, action)]

            # Update Q-value with simulated experience
            self.update_q(state, action, reward, next_state)

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Main learning step: Q-learning + model update + planning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # (a) Q-learning from real experience
        self.update_q(state, action, reward, next_state)

        # (b) Update model
        self.update_model(state, action, reward, next_state)

        # (c) Model-based planning
        self.plan()


def train_dyna_q(
    env: gym.Env,
    agent: DynaQ,
    n_episodes: int = 500
) -> List[float]:
    """
    Train Dyna-Q agent on environment.

    Args:
        env: Gymnasium environment
        agent: DynaQ agent
        n_episodes: Number of training episodes

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Learn from transition
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Planning steps: {agent.n_planning}")

    return episode_rewards


def moving_average(data: List[float], window: int = 50) -> np.ndarray:
    """
    Compute moving average for smoothing learning curves.

    Args:
        data: Input data
        window: Window size

    Returns:
        Smoothed data
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def compare_planning_steps():
    """
    Compare Dyna-Q performance with different numbers of planning steps.
    Demonstrates that more planning leads to faster learning.
    """
    # FrozenLake-v1 (4x4 grid, deterministic)
    env = gym.make('FrozenLake-v1', is_slippery=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    n_episodes = 500

    # Test different planning steps
    planning_configs = [0, 5, 50]  # 0 = pure Q-learning
    results = {}

    print("Training agents with different planning steps...\n")

    for n_planning in planning_configs:
        print(f"\n{'='*60}")
        print(f"Training with n_planning = {n_planning}")
        print(f"{'='*60}")

        # Create agent
        agent = DynaQ(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=0.1,
            gamma=0.95,
            epsilon=0.1,
            n_planning=n_planning
        )

        # Train agent
        rewards = train_dyna_q(env, agent, n_episodes)
        results[n_planning] = rewards

    env.close()

    # Plot learning curves
    plt.figure(figsize=(12, 6))

    for n_planning, rewards in results.items():
        smoothed = moving_average(rewards, window=50)
        label = f'n_planning={n_planning}'
        if n_planning == 0:
            label += ' (Pure Q-learning)'
        plt.plot(smoothed, label=label, linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (50-episode window)', fontsize=12)
    plt.title('Dyna-Q: Effect of Planning Steps on Learning Speed', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Print final performance
    print(f"\n{'='*60}")
    print("Final Performance (last 100 episodes):")
    print(f"{'='*60}")
    for n_planning, rewards in results.items():
        avg_reward = np.mean(rewards[-100:])
        print(f"n_planning={n_planning:2d}: {avg_reward:.3f}")

    plt.savefig('/opt/projects/01_Personal/03_Study/examples/Reinforcement_Learning/dyna_q_comparison.png', dpi=150)
    print(f"\nPlot saved to: dyna_q_comparison.png")
    plt.show()


if __name__ == '__main__':
    compare_planning_steps()
