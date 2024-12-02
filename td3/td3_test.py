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
    num_trails = 1                                              # Number of trials
    episodes_per_trail = 100                                    # Episodes per training trial
    batch_size = 64                                              # Batch size for experience replay

    # Initialize storage for results
    td3_returns = []
    td3_actor_losses = []
    td3_critic_losses = []

    """ 
    Train the TD3 agent for multiple trials and collect performance data.
    """
    for _ in range(num_trails):
        agent = TD3Agent(state_dim, action_dim)
        rewards, actor_losses, critic_losses = agent.train(episodes=episodes_per_trail, batch_size=batch_size)

        td3_returns.append(rewards)
        td3_actor_losses.append(actor_losses)
        td3_critic_losses.append(critic_losses)

    """ 
    Plot the average performance of the TD3 agent across training trials.
    """
    # Plot average returns
    plot_curves([np.array(td3_returns)], ['TD3'], ['b'], 'Return', 'TD3 Returns', smoothing=True)
    plt.savefig(os.path.join(plot_path, "td3_returns.png"))
    plt.close()

    # Plot average actor and critic losses
    plot_curves([np.array(td3_actor_losses), np.array(td3_critic_losses)], ['TD3 Actor Loss', 'TD3 Critic Loss'],
                ['g', 'r'], 'Loss', 'Actor & Critic Loss', smoothing=True)
    plt.savefig(os.path.join(plot_path, "td3_losses.png"))
    plt.close()

    env.close()