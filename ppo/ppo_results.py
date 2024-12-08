import pickle
import matplotlib.pyplot as plt
import numpy as np
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

results_path = os.path.join("training_results")

# Load results
with open(os.path.join(results_path, "ppo_results.pkl"), "rb") as f:
    ppo_results = pickle.load(f)

ppo_returns = ppo_results["ppo_returns"]
ppo_actor_losses = ppo_results["ppo_actor_losses"]
ppo_steps = ppo_results["ppo_steps"]

# Defining the path to store plots
plot_path = os.path.join("plot_figs")

# Create the plot directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

"""
Plot the average performance of the PPO agent across training trials.
"""
#--------------------- Plot average return for ppo
plot_curves([np.array(ppo_returns)], ['PPO'], ['g'], 'Return', 'PPO Returns', smoothing=True)
filepath1 = os.path.join(plot_path, "ppo_returns.png")

# Check if file exists
if os.path.exists(filepath1):
    os.remove(filepath1)

# Save the figure
plt.savefig(filepath1)
plt.close()

#--------------------- Plot average actor loss
plot_curves([np.array(ppo_actor_losses)], ['PPO Actor Loss'],
            ['r'], 'Loss', 'Actor Loss', smoothing=True)
filepath3 = os.path.join(plot_path, "ppo_actor_loss.png")

# Check if file exists
if os.path.exists(filepath3):
    os.remove(filepath3)

# Save the figure
plt.savefig(filepath3)
plt.close()

#--------------------- Plot average steps for each trial
plot_curves([np.array(ppo_steps)], ['Steps'],
            ['k'], 'Steps taken', 'PPO Steps', smoothing=True)
filepath4 = os.path.join(plot_path, "ppo_steps.png")

# Check if file exists
if os.path.exists(filepath4):
    os.remove(filepath4)

# Save the figure
plt.savefig(filepath4)
plt.close()
