# DQN Test script for LunarLander-v3
# Adapted from f_dqn_cartpole_test.py
#
# ── HOW TO SELECT WHICH MODEL TO TEST ──────────────────────────────────────────
# Set MODEL_PATH to the .pth file you want to evaluate, e.g.:
#
#   After training with lr=0.001:
#     MODEL_PATH = "models/dqn_LunarLander-v3_lr_0.001_latest.pth"
#
#   After training with lr=0.00001:
#     MODEL_PATH = "models/dqn_LunarLander-v3_lr_0.00001_latest.pth"
#
# Leave MODEL_PATH = None to fall back to "models/dqn_LunarLander-v3_latest.pth"
# (which no longer exists with the new naming scheme, so always set it explicitly).
# ───────────────────────────────────────────────────────────────────────────────
import gymnasium as gym
import os

from qnet_lunar_lander import QNet
from dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# ── CHANGE THIS LINE to select the model you want to test ──────────────────────
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "dqn_LunarLander-v3_lr_0.001_latest.pth")
# MODEL_PATH = os.path.join(CURRENT_DIR, "models", "dqn_LunarLander-v3_lr_0.00001_latest.pth")
# ───────────────────────────────────────────────────────────────────────────────


def main():
    ENV_NAME = "LunarLander-v3"

    # rgb_array mode required by RecordVideo wrapper
    test_env = gym.make(ENV_NAME, render_mode="rgb_array")

    # Must match the network architecture used during training
    qnet = QNet(n_features=8, n_actions=4)

    dqn_tester = DqnTester(
        env=test_env,
        qnet=qnet,
        env_name=ENV_NAME,
        current_dir=CURRENT_DIR,
        model_path=MODEL_PATH,   # explicit path — no guessing
    )

    dqn_tester.test()
    test_env.close()


if __name__ == "__main__":
    main()
