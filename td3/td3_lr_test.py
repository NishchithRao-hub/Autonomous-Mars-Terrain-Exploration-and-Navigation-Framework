import tqdm
import pickle
from td3_agent import TD3Agent
import gym
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

    results_path = os.path.join("training_results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Initialize the environment
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)  # Initialize the environment
    state_dim = np.prod(env.observation_space.shape)  # Flattened state space (21x21 grid)
    action_dim = env.action_space.n  # 4 discrete actions (up, down, left, right)

    # Training parameters
    num_trials = 10  # Number of trials
    episodes_per_trail = 5000  # Episodes per training trial
    batch_size = 64  # Batch size for experience replay
    learning_rates = [1e-4, 1e-3, 1e-2, 3e-4]

    # Initialize storage for results
    td3_lr_returns = []

    """
    Train the TD3 agent for multiple trials and collect performance data.
    """
    for alpha in learning_rates:
        tr_bar = tqdm.trange(num_trials)
        for trial in tr_bar:
            agent = TD3Agent(state_dim, action_dim, batch_size, learning_rate=alpha)
            rewards, actor_losses, critic_losses, steps, area_covered = agent.train(episodes=episodes_per_trail)

            td3_lr_returns.append(rewards)

            # Save results using pickle
            with open(os.path.join(results_path, "td3_lr_results.pkl"), "wb") as f:
                pickle.dump({"td3_lr_results": td3_lr_returns}, f)
            print(f"Results saved to {os.path.join(results_path, 'td3_lr_results.pkl')}")

            tr_bar.set_description(
                f" Average Reward for learning rate: {alpha} is {sum(rewards) / len(rewards):.2f}"
            )

    env.close()

    # Load results
    with open(os.path.join(results_path, "td3_lr_results.pkl"), "rb") as f:
        results = pickle.load(f)

    td3_lr_results = results["td3_lr_results"]

    grouped_returns = [td3_lr_results[i * num_trials: (i + 1) * num_trials] for i in range(len(learning_rates))]

    # Plotting
    plt.figure(figsize=(12, 8))
    for i, (lr, rewards_group) in enumerate(zip(learning_rates, grouped_returns)):
        avg_rewards = np.mean(rewards_group, axis=0)
        plt.plot(avg_rewards, label=f"Learning Rate = {lr:.1e}")

    # Add labels, legend, and title
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.title("TD3 Agent Performance Across Learning Rates", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    filepath1 = os.path.join(plot_path, "td3_lr_results.png")

    # Check if file exists
    if os.path.exists(filepath1):
        os.remove(filepath1)

    # Save the figure
    plt.savefig(filepath1)
    plt.close()
