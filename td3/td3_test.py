
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