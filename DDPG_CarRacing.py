"""
DDPG (Deep Deterministic Policy Gradient) from scratch
Applied to CarRacing-v2 (Gymnasium)
Based on: Lillicrap et al. 2016 - https://arxiv.org/abs/1509.02971
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# ─────────────────────────────────────────────
# 1. REPLAY BUFFER
# Stores past (s, a, r, s', done) transitions.
# Sampling random batches breaks temporal
# correlation — key for stable off-policy learning.
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 2. ACTOR NETWORK
# Maps state → action (deterministic policy).
# Uses CNN to process pixel observations from
# CarRacing-v2, then outputs 3 continuous values:
#   [steering, throttle, brake] ∈ [-1, 1]
# tanh squashes output to valid action range.
# ─────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # CNN feature extractor — processes 96x96 RGB frames
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # → 23x23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # → 10x10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # → 8x8
            nn.ReLU(),
            nn.Flatten()                                   # → 4096
        )
        # MLP head — maps features to action
        self.mlp = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # squash to [-1, 1]
        )

    def forward(self, x):
        # Normalize pixels to [0, 1] and fix channel order
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        return self.mlp(self.cnn(x))


# ─────────────────────────────────────────────
# 3. CRITIC NETWORK
# Maps (state, action) → Q value (scalar).
# Estimates expected future reward of taking
# action a in state s: Q(s, a).
# CNN processes pixels, then concatenates action
# before the final MLP layers.
# ─────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Concatenate action with CNN features
        self.mlp = nn.Sequential(
            nn.Linear(4096 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # single Q value output
        )

    def forward(self, x, action):
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        return self.mlp(torch.cat([features, action], dim=1))


# ─────────────────────────────────────────────
# 4. ORNSTEIN-UHLENBECK NOISE
# Temporally correlated noise for exploration.
# Better than pure Gaussian noise for continuous
# control because it produces smoother sequences.
# Think of it as "momentum" in the noise signal.
# ─────────────────────────────────────────────
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


# ─────────────────────────────────────────────
# 5. DDPG AGENT
# Ties everything together.
# Key mechanisms:
#   - Actor/Critic + their frozen target copies
#   - Polyak averaging for soft target updates
#   - Replay buffer for off-policy learning
# ─────────────────────────────────────────────
class DDPGAgent:
    def __init__(self, action_dim, device, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, batch_size=64):
        self.device = device
        self.gamma = gamma    # discount factor
        self.tau = tau        # polyak averaging rate
        self.batch_size = batch_size

        # Actor + frozen target copy
        self.actor = Actor(action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        # Critic + frozen target copy
        self.critic = Critic(action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)
        self.loss_fn = nn.MSELoss()

    def select_action(self, obs, explore=True):
        # Convert obs to tensor, get actor's action
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        self.actor.train()
        if explore:
            action += self.noise.sample()  # add OU noise for exploration
        return np.clip(action, -1, 1)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None, None  # wait until buffer has enough samples

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        obs_t      = torch.FloatTensor(obs).to(self.device)
        actions_t  = torch.FloatTensor(actions).to(self.device)
        rewards_t  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ── Critic Update ──────────────────────────
        # TD target: r + γ * Q_target(s', π_target(s'))
        # This is the Bellman equation — what Q(s,a)
        # *should* be if the critic were perfect.
        # We stop gradients through the target (detach)
        # so we're not chasing a moving target.
        with torch.no_grad():
            next_actions = self.actor_target(next_obs_t)
            q_target = rewards_t + self.gamma * (1 - dones_t) * \
                       self.critic_target(next_obs_t, next_actions)

        q_current = self.critic(obs_t, actions_t)
        critic_loss = self.loss_fn(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor Update ───────────────────────────
        # Maximize Q(s, π(s)) — train actor to output
        # actions the critic scores highly.
        # Negative because optimizers minimize.
        actor_loss = -self.critic(obs_t, self.actor(obs_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Soft Target Update (Polyak Averaging) ──
        # Slowly blend target networks toward main networks.
        # τ=0.005 means target moves 0.5% each step.
        # This stability trick prevents oscillating updates.
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, main, target):
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("CarRacing-v2", continuous=True)
    action_dim = env.action_space.shape[0]  # 3: steer, throttle, brake

    agent = DDPGAgent(action_dim, device)
    writer = SummaryWriter("./tb_logs/DDPG")

    max_episodes = 50
    max_steps = 500
    warmup_steps = 500  # fill buffer before training
    total_steps = 0

    for episode in range(max_episodes):
        obs, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        critic_losses, actor_losses = [], []

        for step in range(max_steps):
            # During warmup: random actions to fill buffer
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, explore=True)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            episode_reward += reward
            total_steps += 1

            # Train after warmup
            if total_steps >= warmup_steps:
                c_loss, a_loss = agent.train()
                if c_loss is not None:
                    critic_losses.append(c_loss)
                    actor_losses.append(a_loss)

            if done:
                break

        # Log to TensorBoard
        writer.add_scalar("rollout/ep_rew_mean", episode_reward, total_steps)
        if critic_losses:
            writer.add_scalar("train/critic_loss", np.mean(critic_losses), total_steps)
            writer.add_scalar("train/actor_loss", np.mean(actor_losses), total_steps)

        print(f"Episode {episode} | Steps {total_steps} | Reward {episode_reward:.1f}")

    env.close()
    writer.close()


if __name__ == "__main__":
    train()
