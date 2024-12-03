import tqdm
from td3_agent import TD3Agent
import gym
import matplotlib.pyplot as plt
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def moving_average(data, *, window_size = 50):
    """Smooths 1-D data array using a moving average.
    Args:
        data: 1-D numpy.array
        window_size: Size of the smoothing window

    Returns:
        smooth_data: A 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title, smoothing = True):
    """
    Args:
        arr_list (list): List of results arrays to plot
        legend_list (list): List of legends corresponding to each result array
        color_list (list): List of color corresponding to each result array
        ylabel (string): Label of the vertical axis

        Make sure the elements in the arr_list, legend_list, and color_list
        are associated with each other correctly (in the same order).
        Do not forget to change the ylabel for different plots.
    """
    # Set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    # Plot results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # Compute the standard error (of raw data, not smoothed)
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # Plot the mean
        averages = moving_average(arr.mean(axis=0)) if smoothing else arr.mean(axis=0)
        h, = ax.plot(range(arr.shape[1]), averages, color=color, label=legend)
        # Plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), averages - arr_err, averages + arr_err, alpha=0.3,
                        color=color)
        # Save the plot handle
        h_list.append(h)

    # Plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)


if __name__ == "__main__":
    """
    Main script to train and evaluate the TD3 agent on the Mars Explorer environment.

    This script defines the training parameters, initializes the environment and agent,
    runs multiple training trials, and plots the average performance of the agent.
    """

    # Defining the path to store plots
    plot_path = os.path.join("plot_figs")

    # Create the plot directory if it doesn't exist
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Initialize the environment
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)   # Initialize the environment
    state_dim = np.prod(env.observation_space.shape)             # Flattened state space (21x21 grid)
    action_dim =env.action_space.n                               # 4 discrete actions (up, down, left, right)

    # Training parameters
    num_trails = 3                                               # Number of trials
    episodes_per_trail = 100                                     # Episodes per training trial
    batch_size = 64                                              # Batch size for experience replay

    # Initialize storage for results
    td3_returns = []
    td3_actor_losses = []
    td3_critic_losses = []

    """ 
    Train the TD3 agent for multiple trials and collect performance data.
    """
    tr_bar = tqdm.trange(num_trails)
    for trial in tr_bar:
        agent = TD3Agent(state_dim, action_dim, batch_size)
        rewards, actor_losses, critic_losses = agent.train(episodes=episodes_per_trail)

        td3_returns.append(rewards)
        td3_actor_losses.append(actor_losses)
        td3_critic_losses.append(critic_losses)

        tr_bar.set_description(
            f" Average Reward: {sum(rewards)/len(rewards):.2f} | Average Critic Loss: {sum(critic_losses)/len(critic_losses):.2f} | "
            f"Average Actor Loss: {sum(actor_losses)/len(actor_losses):.2f}"
        )

    """ 
    Plot the average performance of the TD3 agent across training trials.
    """
    # Plot average returns
    plot_curves([np.array(td3_returns)], ['TD3'], ['g'], 'Return', 'TD3 Returns', smoothing=True)
    filepath1 = os.path.join(plot_path, "td3_returns.png")

    # Check if file exists
    if os.path.exists(filepath1):
        os.remove(filepath1)

    # Save the figure
    plt.savefig(filepath1)
    plt.close()

    # Plot average actor and critic losses
    plot_curves([np.array(td3_critic_losses)], ['TD3 Critic Loss'],
                ['b'], 'Loss', 'Critic Loss', smoothing=True)
    filepath2 = os.path.join(plot_path, "td3_critic_loss.png")

    # Check if file exists
    if os.path.exists(filepath2):
        os.remove(filepath2)

    # Save the figure
    plt.savefig(filepath2)
    plt.close()

    # Plot average actor and critic losses
    plot_curves([np.array(td3_actor_losses)], ['TD3 Actor Loss'],
                ['r'], 'Loss', 'Actor Loss', smoothing=True)
    filepath3 = os.path.join(plot_path, "td3_actor_loss.png")

    # Check if file exists
    if os.path.exists(filepath3):
        os.remove(filepath3)

    # Save the figure
    plt.savefig(filepath3)
    plt.close()

    env.close()