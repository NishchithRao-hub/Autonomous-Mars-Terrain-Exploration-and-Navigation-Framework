import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space, hidden_sizes):
        super(MLPPolicy, self).__init__()
        # self.obs_space = obs_space
        self.action_space = action_space

        if isinstance(obs_space, gym.spaces.Box):
            in_dim = np.prod(obs_space.shape[0])  # For Box space, we can safely access shape
        else:
            raise ValueError("Expected obs_space to be of type gym.spaces.Box, but got {}".format(type(obs_space)))

        # in_dim = obs_space.shape[0]
        # Define the layers
        self.actor = self.build_mlp(in_dim, action_space, hidden_sizes)
        self.critic = self.build_mlp(in_dim, 1, hidden_sizes)

    def build_mlp(self, input_dim, output_dim, hidden_sizes):
        layers = []
        prev_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_dim))
        return nn.Sequential(*layers)

    def forward(self, obs):
        # Forward pass for the actor (policy) and critic (value function)
        obs = obs.view(obs.size(0), -1)
        action_probs = self.actor(obs)
        value = self.critic(obs)
        return action_probs, value

    def act(self, obs):
        # Sample an action from the policy
        obs = obs.view(1, -1)
        action_probs, _ = self.forward(obs)
        action = torch.argmax(action_probs, dim=-1).item()  # Choose the action with the highest probability
        return action
