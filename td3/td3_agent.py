import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer
from actor_critic import Actor, Critic

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device,
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
                 noise_std=0.2, noise_clip=0.5, policy_delay=2):
        self.device = device

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000)
        self.total_it = 0

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        self.total_it += 1

        # Sample batch
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Compute target Q value
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

        # Update Critic
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed Actor Update
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)
