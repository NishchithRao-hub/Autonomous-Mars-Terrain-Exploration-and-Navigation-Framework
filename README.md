# Reinforcement-Learning-Based-Autonomous-Mars-Terrain-Exploration-and-Navigation-Framework

Efficient autonomous exploration of uncharted terrains, such as the Martian surface, presents significant challenges in robotics and artificial intelligence. This project addresses the task of navigation and mapping in Mars-like environments by leveraging Deep Reinforcement Learning (DRL). Specifically, it explores and evaluates the performance of Twin Delayed Deep Deterministic Policy Gradient (TD3) and Proximal Policy Optimization (PPO) algorithms within the simulated MarsExplorer environment. The findings demonstrate enhanced terrain exploration, effective obstacle avoidance, and high coverage rates under defined constraints.

## Installation

You can install MarsExplorer environment by using the following commands:

1. Clone the repository.
```shell
$ git clone https://github.com/GouriRajesh/Reinforcement-Learning-Based-Autonomous-Mars-Terrain-Exploration-and-Navigation-Framework.git
```
2. Install the package.
```shell
$ pip install -e mars-explorer
```
3. Install the dependencies.
```shell
For mac users:
$ sh setup.sh

For Windows users:
$ bash setup.sh
```
## Dependancies

You can have a better look at the dependencies at:
```shell
setup/environment.yml
```
## Testing

Please run the following command to make sure that everything works as expected:

```shell
$ python mars-explorer/tests/test.py
```

## Manual Control

We have included a manual control of the agent, via the corresponding arrow keys. Run the manual control environment via:

```shell
$ python mars-explorer/tests/manual.py
```

## Execution of Algorithms

To train your own agents use the below commands.

For TD3:
```shell
$ python td3/td3_train.py
```
For PPO:
```shell
$ python ppo/ppo_train.py
```
For Baseline DQN:
```shell
$ python td3/baseline_dqn_train.py
```
For Baseline PPO:
```shell
$ python ppo/baseline_ppo_train.py
```
All of the results will be located in pickle files at:
```
~/training_results
```
The trained models are then saved as a .pth file at:
```
~/trained_models
```
## Result Plots

To view the results of your training run the below commands:

| Serial No. | Results                                    | Command                   |
|------------|--------------------------------------------|---------------------------|
| 1.         | TD3 Rewards, Actor & Critic Loss           | python td3/td3_results.py |
| 2.         | TD3 Percentage Area Covered                | python td3/td3_percentage_area_covered_result.py |
| 3.         | TD3 vs Baseline DQN                        | python td3/td3_vs_baseline_dqn_result.py |
| 4.         | PPO Rewards, Actor Loss                    | python ppo/ppo_results.py |
| 5.         | PPO Percentage Area Covered                | python ppo/ppo_percentage_area_covered_result.py |
| 6.         | PPO vs Baseline PPO                        | python ppo/ppo_vs_baseline_ppo_result.py |
| 7.         | PPO vs TD3                                 | python ppo/ppo_vs_td3_result.py |
| 8.         | PPO vs TD3 vs Baseline PPO vs Baseline DQN | python td3/ppo_bppo_td3_dqn_comparison_result.py |

To train and view the results of different learning rates use the below commands.

For TD3:
```shell
$ python td3/td3_lr_test.py
```
For PPO:
```shell
$ python ppo/ppo_lr_test.py
```
All the results for the plot functions as described above are stored at:
```
~/plot_figs
```
## Training Simulation

Here is a small video of how the PPO training looks like:

<img src="utils/Mars-Explorer-V1.gif" width="400" height="400">