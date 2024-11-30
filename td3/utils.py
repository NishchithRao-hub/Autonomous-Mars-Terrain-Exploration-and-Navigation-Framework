import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility in numpy, torch, and random.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def soft_update(target_model, source_model, tau: float):
    """
    Perform a soft update of the target model parameters.
    θ_target = τ * θ_source + (1 - τ) * θ_target
    Args:
        target_model: Target PyTorch model to update
        source_model: Source PyTorch model to copy weights from
        tau: Interpolation factor (0 < τ <= 1)
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def hard_update(target_model, source_model):
    """
    Perform a hard update of the target model parameters (copy weights directly).
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.data)

def save_model(agent, directory: str, prefix: str):
    """
    Save the TD3 agent's actor and critic models to the specified directory.
    Args:
        agent: TD3Agent object containing actor and critic models.
        directory: Path to save the models.
        prefix: Prefix for the model filenames.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(agent.actor.state_dict(), os.path.join(directory, f"{prefix}_actor.pth"))
    torch.save(agent.critic1.state_dict(), os.path.join(directory, f"{prefix}_critic1.pth"))
    torch.save(agent.critic2.state_dict(), os.path.join(directory, f"{prefix}_critic2.pth"))
    print(f"Models saved to {directory} with prefix '{prefix}'.")

def load_model(agent, directory: str, prefix: str, device: str = 'cpu'):
    """
    Load the TD3 agent's actor and critic models from the specified directory.
    Args:
        agent: TD3Agent object to load the weights into.
        directory: Path to load the models from.
        prefix: Prefix for the model filenames.
        device: Device on which to load the models ('cpu' or 'cuda').
    """
    actor_path = os.path.join(directory, f"{prefix}_actor.pth")
    critic1_path = os.path.join(directory, f"{prefix}_critic1.pth")
    critic2_path = os.path.join(directory, f"{prefix}_critic2.pth")

    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
    agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
    print(f"Models loaded from {directory} with prefix '{prefix}'.")


def plot_trajectory(visited_positions, grid_size, title="Agent Trajectory"):
    """
    Visualizes the trajectory of the agent on the grid.

    Args:
        visited_positions (set): A set of (x, y) tuples representing visited grid positions.
        grid_size (tuple): The size of the grid as (width, height).
        title (str): Title of the plot.
    """
    grid = np.zeros(grid_size)

    # Mark visited positions
    for x, y in visited_positions:
        grid[y, x] = 1  # Assuming (x, y) indexing

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='hot', interpolation='nearest', origin='upper')
    plt.title(title)
    plt.colorbar(label="Visited (1: Yes, 0: No)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xticks(np.arange(grid_size[0]))
    plt.yticks(np.arange(grid_size[1]))
    plt.grid(color='white', linestyle='-', linewidth=0.5)
    plt.show()
