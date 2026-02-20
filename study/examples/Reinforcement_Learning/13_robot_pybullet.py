#!/usr/bin/env python3
"""
Robot Control using PyBullet Simulation with PPO

This script demonstrates reinforcement learning for robot control using PyBullet.
We'll train a Kuka robotic arm to reach a target position using Proximal Policy
Optimization (PPO).

If PyBullet is not available, we fall back to a custom cartpole environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import sys

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("PyBullet not available. Using custom fallback environment.")


@dataclass
class Transition:
    """Store a single transition in the environment."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous control.

    Actor outputs mean and log_std for a Gaussian policy.
    Critic outputs state value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Returns:
            action_mean: Mean of action distribution
            action_std: Standard deviation of action distribution
            value: State value estimate
        """
        features = self.shared(state)

        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, action_std, value = self(state)

        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


class KukaReachEnv:
    """Kuka robot arm reaching task using PyBullet."""

    def __init__(self, gui: bool = False):
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load environment
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Robot parameters
        self.num_joints = 7  # Kuka IIWA has 7 DoF
        self.end_effector_index = 6

        # Target position
        self.target_pos = None
        self.target_visual = None

        self._setup_joints()

    def _setup_joints(self):
        """Setup joint parameters."""
        self.joint_indices = list(range(self.num_joints))

        # Get joint limits
        self.joint_lower = []
        self.joint_upper = []
        for i in self.joint_indices:
            info = p.getJointInfo(self.robot, i)
            self.joint_lower.append(info[8])
            self.joint_upper.append(info[9])

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset joint positions to random configuration
        for i in self.joint_indices:
            pos = np.random.uniform(self.joint_lower[i], self.joint_upper[i])
            p.resetJointState(self.robot, i, pos)

        # Random target position
        self.target_pos = np.array([
            np.random.uniform(0.3, 0.6),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.3, 0.6)
        ])

        # Create visual marker for target
        if self.target_visual is not None:
            p.removeBody(self.target_visual)

        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]
        )
        self.target_visual = p.createMultiBody(
            baseVisualShapeIndex=self.target_visual,
            basePosition=self.target_pos
        )

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions and velocities
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        joint_pos = np.array([state[0] for state in joint_states])
        joint_vel = np.array([state[1] for state in joint_states])

        # End effector position
        ee_state = p.getLinkState(self.robot, self.end_effector_index)
        ee_pos = np.array(ee_state[0])

        # Distance to target
        distance = ee_pos - self.target_pos

        obs = np.concatenate([joint_pos, joint_vel, distance])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return next state, reward, done."""
        # Apply action as joint velocities
        action = np.clip(action, -1, 1)

        for i, idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot, idx,
                p.VELOCITY_CONTROL,
                targetVelocity=action[i] * 2.0,  # Scale velocity
                force=100
            )

        p.stepSimulation()

        # Get new observation
        obs = self._get_obs()

        # Compute reward
        ee_state = p.getLinkState(self.robot, self.end_effector_index)
        ee_pos = np.array(ee_state[0])
        distance = np.linalg.norm(ee_pos - self.target_pos)

        reward = -distance  # Negative distance as reward

        # Success bonus
        if distance < 0.05:
            reward += 10.0
            done = True
        else:
            done = False

        return obs, reward, done

    def close(self):
        """Cleanup environment."""
        p.disconnect()


class FallbackCartPoleEnv:
    """Simple CartPole environment as fallback when PyBullet is not available."""

    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5
        self.dt = 0.02

        self.state = None

    def reset(self) -> np.ndarray:
        """Reset to initial state."""
        self.state = np.random.randn(4) * 0.1
        return self.state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action."""
        force = action[0] * 10.0  # Scale action

        x, x_dot, theta, theta_dot = self.state

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Physics equations
        temp = (force + self.mass_pole * self.length * theta_dot**2 * sin_theta) / (
            self.mass_cart + self.mass_pole
        )
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0/3.0 - self.mass_pole * cos_theta**2 /
                          (self.mass_cart + self.mass_pole))
        )
        x_acc = temp - self.mass_pole * self.length * theta_acc * cos_theta / (
            self.mass_cart + self.mass_pole
        )

        # Update state
        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Reward and done
        done = abs(x) > 2.4 or abs(theta) > 0.2
        reward = 1.0 if not done else 0.0

        return self.state.astype(np.float32), reward, done


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory: List[Transition] = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)

        return action.numpy()[0], log_prob.item(), value.item()

    def store_transition(self, transition: Transition):
        """Store transition in memory."""
        self.memory.append(transition)

    def update(self):
        """Update policy using PPO."""
        # Convert memory to tensors
        states = torch.FloatTensor([t.state for t in self.memory])
        actions = torch.FloatTensor([t.action for t in self.memory])
        old_log_probs = torch.FloatTensor([t.log_prob for t in self.memory])

        # Compute returns and advantages
        returns = []
        advantages = []
        running_return = 0
        running_advantage = 0

        for t in reversed(self.memory):
            running_return = t.reward + self.gamma * running_return * (1 - t.done)
            returns.insert(0, running_return)

            td_error = t.reward + self.gamma * running_return * (1 - t.done) - t.value
            running_advantage = td_error + self.gamma * 0.95 * running_advantage * (1 - t.done)
            advantages.insert(0, running_advantage)

        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Evaluate actions
            action_mean, action_std, values = self.policy(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            # Compute ratio and surrogate loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()


def train_robot(episodes: int = 500, gui: bool = False):
    """Train robot using PPO."""
    # Create environment
    if PYBULLET_AVAILABLE:
        env = KukaReachEnv(gui=gui)
        state_dim = 17  # 7 joint pos + 7 joint vel + 3 distance
        action_dim = 7
        print("Using PyBullet Kuka environment")
    else:
        env = FallbackCartPoleEnv()
        state_dim = 4
        action_dim = 1
        print("Using fallback CartPole environment")

    agent = PPOAgent(state_dim, action_dim)

    episode_rewards = []
    update_freq = 10  # Update every N episodes

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(200):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)

            transition = Transition(
                state, action, reward, next_state, done, log_prob, value
            )
            agent.store_transition(transition)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        # Update policy
        if (episode + 1) % update_freq == 0:
            agent.update()

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    if PYBULLET_AVAILABLE:
        env.close()

    print(f"\nTraining completed! Final average reward: {np.mean(episode_rewards[-100:]):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robot control with PPO")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")

    args = parser.parse_args()

    train_robot(episodes=args.episodes, gui=args.gui)
