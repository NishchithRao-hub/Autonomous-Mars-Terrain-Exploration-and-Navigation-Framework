import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        """
        Initialize the Replay Buffer.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_size (int): Maximum size of the buffer.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Allocate memory for the buffer
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        Args:
            state (np.array): Current state.
            action (np.array): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state after taking the action.
            done (bool): Whether the episode is done.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            dict: A dictionary containing sampled states, actions, rewards, next states, and done flags.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "state": torch.FloatTensor(self.state[idxs]),
            "action": torch.FloatTensor(self.action[idxs]),
            "reward": torch.FloatTensor(self.reward[idxs]),
            "next_state": torch.FloatTensor(self.next_state[idxs]),
            "done": torch.FloatTensor(self.done[idxs]),
        }
        return batch
