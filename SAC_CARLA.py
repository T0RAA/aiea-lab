"""
SAC (Soft Actor-Critic) from scratch
Applied to CARLA Gymnasium environment
Based on: Haarnoja et al. 2018 - https://arxiv.org/abs/1801.01290
"""

import gymnasium as gym
#from gymnasium.envs.registration import register
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
#import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"

# ─────────────────────────────────────────────
# 1. REPLAY BUFFER (same as DDPG)
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
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 2. GAUSSIAN ACTOR
# KEY DIFFERENCE FROM DDPG:
# Instead of outputting a single deterministic
# action, SAC's actor outputs mean (μ) and
# log_std (σ) of a Gaussian distribution.
# Actions are sampled via reparameterization:
#   a = μ + σ·ε,  ε ~ N(0,1)
# This makes the policy stochastic — exploration
# is built in, no need for external noise.
# tanh squashes to valid action range.
# ─────────────────────────────────────────────
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, x):
        features = self.mlp(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        # Reparameterization trick: sample from N(mean, std)
        normal = Normal(mean, std)
        x_t = normal.rsample()          # rsample = reparameterized sample
        action = torch.tanh(x_t)        # squash to [-1, 1]

        # Compute log probability with tanh correction
        # (required for entropy calculation)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# ─────────────────────────────────────────────
# 3. DOUBLE CRITIC
# KEY DIFFERENCE FROM DDPG:
# SAC uses TWO Q-networks and takes the MINIMUM
# of their predictions for the TD target.
# This prevents Q-value overestimation which
# causes policy collapse in DDPG.
# ─────────────────────────────────────────────
class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 — independent network, same architecture
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


# ─────────────────────────────────────────────
# 4. SAC AGENT
# KEY DIFFERENCE FROM DDPG:
# Auto-tuning of temperature α.
# α controls exploration/exploitation tradeoff.
# Instead of manually tuning it, SAC treats it
# as a learnable parameter with its own loss:
#   L(α) = α · (-log π(a|s) - target_entropy)
# target_entropy = -action_dim (heuristic)
# ─────────────────────────────────────────────
class SACAgent:
    def __init__(self, obs_dim, action_dim, device,
                 lr=3e-4, gamma=0.99, tau=0.005,
                 batch_size=64, auto_alpha=True):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha

        # Actor
        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Double Critic + target
        self.critic = DoubleCritic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Auto temperature α
        if auto_alpha:
            self.target_entropy = -action_dim  # heuristic from paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = 0.2  # fixed temperature

        self.buffer = ReplayBuffer()

    def select_action(self, obs, evaluate=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if evaluate:
            # Deterministic action for evaluation
            mean, _ = self.actor(obs_t)
            action = torch.tanh(mean)
        else:
            # Stochastic action for training
            action, _ = self.actor.sample(obs_t)
        return action.detach().cpu().numpy()[0]

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        actions_t  = torch.FloatTensor(actions).to(self.device)
        rewards_t  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ── Critic Update ───────────────────────
        # Soft Bellman target includes entropy term:
        # y = r + γ·(min(Q1,Q2)(s',a') - α·log π(a'|s'))
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs_t)
            q1_next, q2_next = self.critic_target(next_obs_t, next_actions)
            q_next_min = torch.min(q1_next, q2_next)
            # Entropy term subtracts α·log_prob from target
            q_target = rewards_t + self.gamma * (1 - dones_t) * \
                       (q_next_min - self.alpha * next_log_prob)

        q1_curr, q2_curr = self.critic(obs_t, actions_t)
        critic_loss = F.mse_loss(q1_curr, q_target) + \
                      F.mse_loss(q2_curr, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor Update ────────────────────────
        # Maximize: E[min(Q1,Q2)(s,a) - α·log π(a|s)]
        actions_new, log_prob = self.actor.sample(obs_t)
        q1_new, q2_new = self.critic(obs_t, actions_new)
        q_new_min = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha (Temperature) Update ──────────
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (
                log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)

        # ── Soft Target Update (Polyak) ─────────
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item(), \
               alpha_loss.item() if self.auto_alpha else 0.0


# ─────────────────────────────────────────────
# 5. TRAINING LOOP — CARLA
# ─────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CARLA params
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,          # SAC needs continuous actions
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 4000,
        'town': 'Town03',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': True,
    }

    #register(id="carla-v0", entry_point="gym_carla.envs:CarlaEnv")
    #env = gym.make("carla-v0", params=params, disable_env_checker=True)
    env = gym.make("CarRacing-v2", continuous=True)


    # Flatten observation for MLP
    obs, _ = env.reset()

    #obs = obs[:, ::4, ::4, :]

    obs_flat = obs.flatten()
    obs_dim = obs_flat.shape[0] # 96 x 96 x 3 = 27,648



    action_dim = env.action_space.shape[0]  # continuous: [accel, steer]

    print(f"obs_dim: {obs_dim}, action_dim: {action_dim}")

    print(f"obs_dim: {obs_dim}, action_dim: {action_dim}")

    agent = SACAgent(obs_dim, action_dim, device)
    writer = SummaryWriter("./tb_logs/SAC_CARLA")

    max_episodes = 500
    max_steps = 1000
    warmup_steps = 1000
    total_steps = 0

    for episode in range(max_episodes):
        obs, _ = env.reset()
        obs = obs.flatten()
        episode_reward = 0
        critic_losses, actor_losses, alphas = [], [], []

        for step in range(max_steps):
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            #next_obs = next_obs[:, ::4, ::4, :]
            next_obs = next_obs.flatten()
            done = terminated or truncated

            agent.buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            episode_reward += reward
            total_steps += 1

            if total_steps >= warmup_steps:
                result = agent.train()
                if result[0] is not None:
                    critic_losses.append(result[0])
                    actor_losses.append(result[1])
                    alphas.append(agent.alpha)

            if done:
                break

        writer.add_scalar("rollout/ep_rew_mean", episode_reward, total_steps)
        if critic_losses:
            writer.add_scalar("train/critic_loss", np.mean(critic_losses), total_steps)
            writer.add_scalar("train/actor_loss", np.mean(actor_losses), total_steps)
            writer.add_scalar("train/alpha", np.mean(alphas), total_steps)

        print(f"Ep {episode} | Steps {total_steps} | "
              f"Reward {episode_reward:.1f} | α {agent.alpha:.4f}")

    env.close()
    writer.close()


if __name__ == "__main__":
    train()
