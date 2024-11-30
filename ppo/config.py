config = {
    'size': [21, 21],
    'hidden_sizes': [64, 64],        # Number of hidden layers and their sizes
    'learning_rate': 3e-4,          # Learning rate for the optimizer
    'gamma': 0.99,                  # Discount factor for future rewards
    'lambda': 0.95,                 # GAE lambda parameter
    'clip_ratio': 0.2,              # Clip ratio for PPO
    'target_kl': 0.01,              # Target KL divergence for early stopping
    'max_epochs': 10,               # Number of epochs for each update
    'max_steps': 400,               # Maximum steps per episode
    'batch_size': 64,               # Batch size for updating the policy
    'num_episodes': 10,           # Number of episodes for training
}

