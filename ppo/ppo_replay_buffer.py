import numpy as np
import torch

class PPOReplayBuffer:
    """
    Replay buffer for PPO that stores transitions and provides batch sampling for updates.

    Args:
        buffer_size (int): Maximum number of transitions to store in the buffer.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
    """
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.log_prob_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.value_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.advantage_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.return_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size

    def store(self, state, action, log_prob, reward, done, value):
        """
        Stores a single transition in the buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
            done (float): Done signal (1 if episode ends, 0 otherwise).
            value (float): Value estimate for the state.
        """
        if self.ptr >= self.max_size:
            raise ValueError("Replay buffer overflow. Increase buffer size.")

        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.log_prob_buffer[self.ptr] = log_prob
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.value_buffer[self.ptr] = value
        self.ptr += 1

    def compute_advantages_and_returns(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Args:
            last_value (float): Value estimate of the final state.
            gamma (float): Discount factor.
            lam (float): GAE parameter.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        # Compute GAE-Lambda advantage
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = self.discount_sum(deltas, gamma * lam)

        # Compute discounted returns
        self.return_buffer[path_slice] = self.discount_sum(rewards, gamma)[:-1]
        self.path_start_idx = self.ptr

    def discount_sum(self, x, discount):
        """
        Compute the discounted cumulative sum of rewards.
        """
        return np.array([sum(discount ** t * x[t] for t in range(len(x) - i)) for i in range(len(x))])

    def get(self):
        """
        Get all data from the buffer and normalize advantages.

        Returns:
            data (dict): Dictionary containing states, actions, log_probs, advantages, and returns.
        """
        assert self.ptr == self.max_size  # Ensure buffer is full
        self.ptr, self.path_start_idx = 0, 0

        # Normalize advantages
        adv_mean, adv_std = np.mean(self.advantage_buffer), np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / (adv_std + 1e-8)

        data = {
            'states': torch.tensor(self.state_buffer, dtype=torch.float32),
            'actions': torch.tensor(self.action_buffer, dtype=torch.float32),
            'log_probs': torch.tensor(self.log_prob_buffer, dtype=torch.float32),
            'advantages': torch.tensor(self.advantage_buffer, dtype=torch.float32),
            'returns': torch.tensor(self.return_buffer, dtype=torch.float32),
        }
        return data

    