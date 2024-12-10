import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from ppo_results import plot_curves

# Defining the path to store plots
plot_path = os.path.join("plot_figs")

# Create the plot directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

results_path = os.path.join("training_results")

# Load results
with open(os.path.join(results_path, "td3_results.pkl"), "rb") as f:
    td3_results = pickle.load(f)
td3_returns = td3_results["td3_returns"]

# Load results
with open(os.path.join(results_path, "ppo_results.pkl"), "rb") as f:
    ppo_results = pickle.load(f)
ppo_returns = ppo_results["ppo_returns"]

# Convert to numpy array
td3_returns = np.array(td3_returns)
ppo_returns = np.array(ppo_returns)

"""
Plot the average performance of the TD3 vs PPO across trials.
"""
# --------------------- Plot average return for td3 vs ppo
plot_curves([np.array(td3_returns), np.array(ppo_returns)],
            ['TD3', 'PPO'], ['blue', 'green', ], 'Return',
            'TD3 vs PPO Returns', smoothing=True)
filepath1 = os.path.join(plot_path, "td3_vs_ppo_returns.png")

# Check if file exists
if os.path.exists(filepath1):
    os.remove(filepath1)

# Save the figure
plt.savefig(filepath1)
plt.close()
