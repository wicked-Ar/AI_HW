# DQN Training script for LunarLander-v3
# Adapted from e_dqn_cartpole_train.py
#
# Key changes from CartPole:
#   - ENV_NAME = "LunarLander-v3"
#   - QNet(n_features=8, n_actions=4)  [CartPole used n_features=4, n_actions=2]
#   - episode_reward_avg_solved = 200  [fixed by assignment]
#   - validation_num_episodes = 3      [fixed by assignment]
#   - max_num_episodes = 3000          [LunarLander needs more episodes than CartPole]
#   - batch_size = 64                  [larger batch for more stable gradient with LunarLander]
#   - epsilon_start = 1.0              [start with full exploration]
#
# To compare learning rates, change the learning_rate value below and re-run:
#   Experiment 1: "learning_rate": 0.001    (default below)
#   Experiment 2: "learning_rate": 0.00001
# Each run will appear as a separate line in WandB under project "DQN_LunarLander-v3"
import gymnasium as gym
import torch
import os

from qnet_lunar_lander import QNet
from dqn_train_test import DqnTrainer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    print("TORCH VERSION:", torch.__version__)

    ENV_NAME = "LunarLander-v3"

    env = gym.make(ENV_NAME)
    valid_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,
        "max_num_episodes": 3000,
        "batch_size": 64,
        "learning_rate": 0.001,           # EXPERIMENT VARIABLE: change to 0.00001 for 2nd run
        "gamma": 0.99,
        "steps_between_train": 1,
        "replay_buffer_size": 1_000_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_final_scheduled_percent": 0.75,
        "print_episode_interval": 10,
        "target_sync_time_steps_interval": 500,
        "validation_time_steps_interval": 10_000,
        "validation_num_episodes": 3,     # fixed by assignment — do not change
        "episode_reward_avg_solved": 200, # fixed by assignment — do not change
    }

    # LunarLander: 8-dim state, 4 discrete actions
    qnet = QNet(n_features=8, n_actions=4)
    target_qnet = QNet(n_features=8, n_actions=4)

    use_wandb = True

    dqn = DqnTrainer(
        env=env,
        valid_env=valid_env,
        qnet=qnet,
        target_qnet=target_qnet,
        config=config,
        use_wandb=use_wandb,
        current_dir=CURRENT_DIR,
    )

    dqn.train_loop()


if __name__ == "__main__":
    main()
