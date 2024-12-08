import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ppo_results import plot_curves

# Defining the path to store plots
plot_path = os.path.join("plot_figs")

# Create the plot directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

results_path = os.path.join("training_results")

# Load results
with open(os.path.join(results_path, "baseline_ppo_results.pkl"), "rb") as f:
    ppo_results = pickle.load(f)

baseline_ppo_returns = ppo_results["baseline_ppo_returns"]

# Load results
with open(os.path.join(results_path, "ppo_results.pkl"), "rb") as f:
    ppo_results = pickle.load(f)

ppo_returns = ppo_results["ppo_returns"]

# Convert to numpy array and reshape
baseline_ppo_returns = np.squeeze(np.array(baseline_ppo_returns))
ppo_returns = np.array(ppo_returns)

"""
Plot the average performance of the PPO agent across training trials.
"""
#--------------------- Plot average return for ppo vs baseline ppo
plot_curves([np.array(baseline_ppo_returns), np.array(ppo_returns)], ['Baseline PPO', 'PPO'], ['b', 'r'], 'Return', 'Baseline PPO vs PPO Returns', smoothing=True)
filepath1 = os.path.join(plot_path, "ppo_vs_baseline_ppo_returns.png")

# Check if file exists
if os.path.exists(filepath1):
    os.remove(filepath1)

# Save the figure
plt.savefig(filepath1)
plt.close()