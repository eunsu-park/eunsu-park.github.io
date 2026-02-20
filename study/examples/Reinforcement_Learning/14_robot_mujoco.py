#!/usr/bin/env python3
"""
Robot Control with MuJoCo and Soft Actor-Critic (SAC)

MuJoCo (Multi-Joint dynamics with Contact) is the industry-standard physics
engine for robotics research. It provides accurate simulation of rigid body
dynamics, contacts, and actuator models — making it the benchmark environment
for continuous-control RL research (locomotion, manipulation, whole-body control).

Key concepts demonstrated:
  1. Continuous action spaces: robot joints require real-valued torques, not
     discrete choices. Gaussian policies parameterize mean and std for each
     action dimension.
  2. Soft Actor-Critic (SAC): off-policy Actor-Critic that adds an entropy bonus
     to the reward. Maximizing H(π) alongside return drives exploration and
     prevents premature convergence to suboptimal deterministic policies.
  3. Replay buffer: off-policy algorithms reuse past experience, dramatically
     improving sample efficiency over on-policy methods (PPO, REINFORCE).
  4. Twin Q-networks: two separate critics take the minimum of their predictions,
     reducing the systematic overestimation that destabilises single-critic SAC.

Target environment: HalfCheetah-v4 (or Ant-v4) from Gymnasium.
Fallback: a lightweight custom continuous pendulum when MuJoCo is absent.

Requirements (full):  torch  gymnasium[mujoco]  numpy
Requirements (fallback):  torch  numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Tuple, Optional
import random
import sys

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BUFFER_CAPACITY  = 100_000   # Replay buffer size
BATCH_SIZE       = 256        # Minibatch size for SAC updates
HIDDEN_DIM       = 256        # Hidden units in all networks
GAMMA            = 0.99       # Discount factor
TAU              = 0.005      # Soft target-network update rate
ACTOR_LR         = 3e-4       # Actor learning rate
CRITIC_LR        = 3e-4       # Critic learning rate
ALPHA_LR         = 3e-4       # Entropy coefficient learning rate
LOG_STD_MIN      = -20        # Clamp for numerical stability
LOG_STD_MAX      = 2          # Clamp to prevent too-wide distributions
UPDATE_AFTER     = 1_000      # Warm-up steps before first gradient update
UPDATE_EVERY     = 50         # Gradient steps per environment step interval
TRAINING_EPISODES= 10         # Episodes to run (demo; increase for real training)
MAX_STEPS        = 300        # Max steps per episode
SEED             = 42

# ---------------------------------------------------------------------------
# MuJoCo availability check
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    env_test = gym.make("HalfCheetah-v4")
    env_test.close()
    MUJOCO_AVAILABLE = True
    TARGET_ENV = "HalfCheetah-v4"
except Exception:
    MUJOCO_AVAILABLE = False
    print(
        "MuJoCo / Gymnasium[mujoco] not found.\n"
        "To install:  pip install gymnasium[mujoco]\n"
        "             pip install mujoco\n"
        "Falling back to a custom continuous pendulum environment.\n"
    )


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Circular experience replay buffer for off-policy learning.

    Off-policy algorithms (SAC, DQN, TD3) can reuse transitions collected
    under any policy.  This breaks temporal correlations in the data stream
    and makes gradient updates more statistically efficient.
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class GaussianActor(nn.Module):
    """
    Stochastic actor for continuous action spaces.

    Outputs a squashed Gaussian policy:
      1. Two hidden layers produce mean and log_std.
      2. An action sample z ~ N(mean, std) is passed through tanh to bound
         it to (-1, 1) — matching most MuJoCo action spaces.
      3. The log-probability is corrected for the tanh squashing:
            log π(a|s) = log N(z|s) − Σ log(1 − tanh²(z))

    This squashing trick (Haarnoja et al., 2018) avoids actions that would
    saturate actuator limits while keeping the policy differentiable.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) with tanh squashing applied."""
        features = self.net(state)
        mean     = self.mean_layer(features)
        log_std  = self.log_std_layer(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std      = log_std.exp()

        dist   = torch.distributions.Normal(mean, std)
        z      = dist.rsample()           # reparameterisation for backprop
        action = torch.tanh(z)

        # Squashing correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks (critic) for SAC.

    Two independent Q-functions share no weights.  During the TD target
    computation the *minimum* of Q1 and Q2 is used, which counters the
    positive bias that arises when a single network both selects and
    evaluates actions (Fujimoto et al., 2018).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------
class SACAgent:
    """
    Soft Actor-Critic (SAC) — Haarnoja et al., 2018/2019.

    SAC maximises a temperature-weighted entropy-augmented objective:
        J(π) = Σ_t  E[ r(s,a) + α · H(π(·|s)) ]

    Key advantages over on-policy algorithms (PPO) for robotics:
      • Off-policy: far more sample efficient (critical for physical robots).
      • Entropy regularisation: automatic exploration without ε-greedy schedules.
      • Automatic temperature α: no manual tuning of the exploration weight.
    """

    def __init__(self, state_dim: int, action_dim: int, action_scale: float = 1.0):
        self.action_scale = action_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # Twin critics + target critics
        self.critic        = TwinQNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # Entropy temperature α (learned automatically)
        self.target_entropy = -float(action_dim)          # heuristic: -|A|
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim    = optim.Adam([self.log_alpha], lr=ALPHA_LR)

        self.replay_buffer  = ReplayBuffer()
        self.total_steps    = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action from actor; use mean for deterministic evaluation."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(state_t)
            if deterministic:
                # Use mean (no rsample noise)
                features = self.actor.net(state_t)
                action   = torch.tanh(self.actor.mean_layer(features))
        return (action.cpu().numpy()[0] * self.action_scale)

    # ------------------------------------------------------------------
    def _soft_update(self) -> None:
        """Polyak averaging: θ_target ← τ·θ + (1−τ)·θ_target."""
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

    # ------------------------------------------------------------------
    def update(self) -> Optional[dict]:
        """One gradient step for critic, actor, and temperature."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = (
            x.to(self.device) for x in (states, actions, rewards, next_states, dones)
        )

        # ---- Critic update -------------------------------------------
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + GAMMA * (1 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor update --------------------------------------------
        new_actions, log_probs = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new    = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Temperature (α) update ----------------------------------
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha.item(),
        }

    # ------------------------------------------------------------------
    def step_and_maybe_update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[dict]:
        """Store transition; trigger gradient updates on schedule."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if self.total_steps < UPDATE_AFTER:
            return None

        if self.total_steps % UPDATE_EVERY == 0:
            info = None
            for _ in range(UPDATE_EVERY):
                info = self.update()
            return info
        return None


# ---------------------------------------------------------------------------
# Fallback environment (lightweight continuous pendulum)
# ---------------------------------------------------------------------------
class ContinuousPendulumEnv:
    """
    Simple undamped pendulum with continuous torque input.

    State:  [cos θ, sin θ, θ̇]   (3-dim, matching Gymnasium Pendulum-v1)
    Action: [τ]  ∈ [-1, 1]
    Reward: -(θ² + 0.1·θ̇² + 0.001·τ²)  — penalise distance from upright
    """

    def __init__(self):
        self.max_torque  = 2.0
        self.max_speed   = 8.0
        self.g, self.m, self.l = 10.0, 1.0, 1.0
        self.dt          = 0.05
        self.observation_space_shape = (3,)
        self.action_space_shape      = (1,)
        self.theta = self.theta_dot = 0.0

    def reset(self) -> np.ndarray:
        self.theta     = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1.0, 1.0)
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        torque = float(np.clip(action[0], -1, 1)) * self.max_torque
        theta_dot_new = (
            self.theta_dot
            + (-3 * self.g / (2 * self.l) * np.sin(self.theta + np.pi)
               + 3.0 / (self.m * self.l ** 2) * torque)
            * self.dt
        )
        self.theta_dot = np.clip(theta_dot_new, -self.max_speed, self.max_speed)
        self.theta    += self.theta_dot * self.dt

        reward = -(
            self.angle_normalize(self.theta) ** 2
            + 0.1 * self.theta_dot ** 2
            + 0.001 * torque ** 2
        )
        return self._obs(), reward, False   # pendulum never terminates

    def _obs(self) -> np.ndarray:
        return np.array(
            [np.cos(self.theta), np.sin(self.theta), self.theta_dot],
            dtype=np.float32,
        )

    @staticmethod
    def angle_normalize(x: float) -> float:
        return ((x + np.pi) % (2 * np.pi)) - np.pi


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(episodes: int = TRAINING_EPISODES) -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if MUJOCO_AVAILABLE:
        import gymnasium as gym
        env = gym.make(TARGET_ENV)
        state_dim    = env.observation_space.shape[0]
        action_dim   = env.action_space.shape[0]
        action_scale = float(env.action_space.high[0])
        print(f"Environment : {TARGET_ENV}")
        print(f"  obs dim   : {state_dim}  |  action dim : {action_dim}")
        print(f"  action scale : ±{action_scale}")
    else:
        env          = ContinuousPendulumEnv()
        state_dim    = env.observation_space_shape[0]
        action_dim   = env.action_space_shape[0]
        action_scale = 1.0
        print("Environment : ContinuousPendulum (fallback)")

    agent = SACAgent(state_dim, action_dim, action_scale)
    print(f"Device : {agent.device}\n")
    print(f"{'Episode':>8}  {'Steps':>7}  {'Reward':>10}  {'Alpha':>7}")
    print("-" * 45)

    for ep in range(1, episodes + 1):
        if MUJOCO_AVAILABLE:
            state, _ = env.reset(seed=SEED + ep)
        else:
            state = env.reset()

        ep_reward = 0.0
        last_info: Optional[dict] = None

        for _ in range(MAX_STEPS):
            # Random exploration during warm-up; policy afterwards
            if agent.total_steps < UPDATE_AFTER:
                if MUJOCO_AVAILABLE:
                    action = env.action_space.sample()
                else:
                    action = np.random.uniform(-1, 1, size=(action_dim,)).astype(np.float32)
            else:
                action = agent.select_action(state)

            if MUJOCO_AVAILABLE:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done = env.step(action)

            info = agent.step_and_maybe_update(state, action, reward, next_state, done)
            if info:
                last_info = info

            ep_reward += reward
            state      = next_state
            if done:
                break

        alpha_val = last_info["alpha"] if last_info else float(agent.alpha)
        print(
            f"{ep:>8}  {agent.total_steps:>7}  "
            f"{ep_reward:>10.2f}  {alpha_val:>7.4f}"
        )

    if MUJOCO_AVAILABLE:
        env.close()
    print("\nDone. Increase TRAINING_EPISODES / MAX_STEPS for meaningful learning.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC robot control with MuJoCo")
    parser.add_argument(
        "--episodes", type=int, default=TRAINING_EPISODES,
        help="Number of training episodes"
    )
    args = parser.parse_args()

    train(episodes=args.episodes)
