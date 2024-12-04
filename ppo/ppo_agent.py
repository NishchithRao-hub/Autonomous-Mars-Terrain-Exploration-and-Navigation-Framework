import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers in the neural networks.
        lr (float): Learning rate for both policy and value networks.
        gamma (float): Discount factor.
        clip_eps (float): PPO clip ratio.
        entropy_coeff (float): Coefficient for the entropy loss.
        value_coeff (float): Coefficient for the value loss.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-4, gamma=0.99, clip_eps=0.2,
                 entropy_coeff=0.01, value_coeff=0.5):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        # Policy Network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Value Network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        """
        Selects an action based on the current policy.

        Args:
            state (np.ndarray): Current state of the environment.

        Returns:
            action (int): Selected action.
            log_prob (float): Log probability of the selected action.
            value (float): Value estimate for the current state.
        """
        state_tensor = torch.FloatTensor(state).view(-1, self.policy_net.state_dim).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = Categorical(probs)
        action = dist.sample().detach().numpy()[0]
        log_prob = dist.log_prob(action)
        value = self.value_net(state_tensor)
        return action.item(), log_prob.item(), value.item()

    def compute_loss(self, data):
        """
        Computes the PPO loss for policy and value networks.

        Args:
            data (dict): Batch data containing states, actions, log_probs, advantages, and returns.

        Returns:
            policy_loss (torch.Tensor): Loss for the policy network.
            value_loss (torch.Tensor): Loss for the value network.
            entropy_loss (torch.Tensor): Entropy loss to encourage exploration.
        """
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        advantages = data['advantages']
        returns = data['returns']

        # Forward pass for policy network
        probs = self.policy_net(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy_loss = dist.entropy().mean()

        # Ratio for importance sampling
        ratios = torch.exp(log_probs - old_log_probs)

        # Clipped policy loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_net(states).squeeze()
        value_loss = self.mse_loss(values, returns)

        return policy_loss, value_loss, entropy_loss

    def train(self, replay_buffer, epochs=10, batch_size=64):
        """
        Train the agent using data from the replay buffer.

        Args:
            replay_buffer (PPOReplayBuffer): Buffer storing transitions.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for updates.
        """
        data = replay_buffer.get()

        for _ in range(epochs):
            policy_loss, value_loss, entropy_loss = self.compute_loss(data)

            # Update policy network
            self.policy_optimizer.zero_grad()
            (policy_loss - self.entropy_coeff * entropy_loss).backward()
            self.policy_optimizer.step()

            # Update value network
            self.value_optimizer.zero_grad()
            (self.value_coeff * value_loss).backward()
            self.value_optimizer.step()
