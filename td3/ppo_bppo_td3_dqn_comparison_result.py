import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from td3_results import plot_curves


# Defining the path to store plots
plot_path = os.path.join("plot_figs")

# Create the plot directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

results_path = os.path.join("training_results")

# Load results
with open(os.path.join(results_path, "baseline_dqn_results.pkl"), "rb") as f:
    dqn_results = pickle.load(f)
dqn_returns = dqn_results["baseline_dqn_returns"]

# Load results
with open(os.path.join(results_path, "td3_results.pkl"), "rb") as f:
    td3_results = pickle.load(f)
td3_returns = td3_results["td3_returns"]


# Load results
with open(os.path.join(results_path, "baseline_ppo_results.pkl"), "rb") as f:
    ppo_results = pickle.load(f)
baseline_ppo_returns = ppo_results["baseline_ppo_returns"]

# Load results
with open(os.path.join(results_path, "ppo_results.pkl"), "rb") as f:
    ppo_results = pickle.load(f)
ppo_returns = ppo_results["ppo_returns"]


# Convert to numpy array and reshape
dqn_returns = np.squeeze(np.array(dqn_returns))
baseline_ppo_returns = np.squeeze(np.array(baseline_ppo_returns))
td3_returns = np.array(td3_returns)
ppo_returns = np.array(ppo_returns)


"""
Plot the average performance of the TD3 agent across training trials.
"""
#--------------------- Plot average return for td3 vs dqn
plot_curves([np.array(td3_returns), np.array(dqn_returns), np.array(baseline_ppo_returns), np.array(ppo_returns)],
            ['TD3', 'Baseline DQN', 'Baseline PPO', 'PPO'], ['blue', 'yellow', 'green', 'purple'], 'Return',
            'TD3-Baseline DQN-Baseline PPO-PPO Returns Comparison', smoothing=True)
filepath1 = os.path.join(plot_path, "td3_baselinedqn_baselineppo_ppo_returns_comparison.png")

# Check if file exists
if os.path.exists(filepath1):
    os.remove(filepath1)

# Save the figure
plt.savefig(filepath1)
plt.close()