import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_percentage_area_covered(data, ylabel, xlabel, fig_title, save_path):
    """
    Plots percentage area covered with trial numbers on the x-axis, y-axis scaled from 0 to 100,
    and the percentage values displayed above the markers.

    Args:
        data (list or numpy.array): List of percentage area covered per trial.
        ylabel (str): Label for the vertical axis.
        xlabel (str): Label for the horizontal axis.
        fig_title (str): Title of the figure.
        save_path (str): Path to save the plot.
    """
    trial_numbers = np.arange(1, len(data) + 1)  # Trial numbers (1-based index)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(trial_numbers, data, marker='o', linestyle='-', color='blue', label="Area Covered")

    # Add labels, title, and grid
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(fig_title)
    plt.ylim(0, 100)  # Set y-axis scale from 0 to 100
    plt.xticks(trial_numbers)  # Set x-axis ticks to trial numbers
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Annotate each marker with the corresponding value
    for x, y in zip(trial_numbers, data):
        plt.text(x, y + 2, f"{y:.1f}", ha='center', va='bottom', fontsize=10, color='black')

    # Save the plot
    plt.tight_layout()
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()


# Get the plotting values from saved results
results_path = os.path.join("training_results")
# Load results
with open(os.path.join(results_path, "td3_results.pkl"), "rb") as f:
    td3_results = pickle.load(f)
td3_percentage_area_covered = td3_results["td3_percentage_area_covered"]
# Scale the values by 100
td3_percentage_area_covered_scaled = [t * 100 for t in td3_percentage_area_covered]

# Defining the path to store plots
plot_path = os.path.join("plot_figs")
# Create the plot directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
area_covered_path = os.path.join(plot_path, "td3_percentage_area_covered.png")


# Plot
plot_percentage_area_covered(
    data=td3_percentage_area_covered_scaled,
    ylabel="Percentage Area Covered",
    xlabel="Trial Number",
    fig_title="Percentage Area Covered Per Trial",
    save_path=area_covered_path)