import numpy as np
import random


class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experience tuples.

    Args:
        buffer_size (int): Maximum capacity of the buffer.
        batch_size (int): Number of samples to draw from the buffer in each batch.
    """

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []  # List to store experience tuples
        self.pos = 0  # Index of the next position to add to the buffer

    def add(self, state, action, action_values, reward, next_state, done):
        """
        Adds an experience tuple to the buffer.

        Args:
            state (np.array): Current state of the environment.
            action (np.array): Action taken in the current state.
            action_values (np.array): Action values predicted by the actor network.
            reward (float): Reward received from the environment.
            next_state (np.array): Next state of the environment.
            done (bool): Whether the episode has terminated.
        """

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, action_values, reward, next_state, done)
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self):
        """
        Samples a batch of experience tuples from the buffer.

        Returns:
            tuple: A tuple of numpy arrays containing states, actions, action values, rewards, next states, and done
            flags.
        """

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, action_values, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(action_values),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

    def size(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)
