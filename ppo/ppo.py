import torch
import torch.optim as optim
import torch.nn.functional as F
from model import MLPPolicy
import numpy as np


class PPO:
    def __init__(self, obs_space, action_space, config):
        self.obs_space = np.prod(obs_space.shape)
        # self.obs_space = obs_space[0] * obs_space[1] * obs_space[2]
        self.action_space = action_space
        self.hidden_sizes = config['hidden_sizes']
        self.lr = config['learning_rate']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.clip_ratio = config['clip_ratio']
        self.target_kl = config['target_kl']
        self.max_epochs = config['max_epochs']

        self.policy = MLPPolicy(obs_space, action_space, self.hidden_sizes)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def compute_gae(self, rewards, values, dones, next_values):
        # Compute Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (1 - dones[t]) * self.gamma * next_values[t] - values[t]
            advantages[t] = last_advantage = delta + (1 - dones[t]) * self.gamma * self.lam * last_advantage
        return advantages

    def update(self, obs, actions, rewards, values, dones, next_values):
        advantages = self.compute_gae(rewards, values, dones, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.max_epochs):
            action_probs, value_preds = self.policy(obs)

            # Compute the ratio between new and old action probabilities
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
            old_action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))  # Old probabilities

            ratio = torch.exp(action_log_probs - old_action_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

            # Critic loss
            value_loss = F.mse_loss(value_preds, rewards + self.gamma * next_values * (1 - dones))

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Early stopping if KL divergence exceeds target
            approx_kl = (old_action_log_probs - action_log_probs).mean().item()
            if approx_kl > self.target_kl:
                print(f"Early stopping due to KL divergence: {approx_kl}")
                break

        return policy_loss.item(), value_loss.item()
