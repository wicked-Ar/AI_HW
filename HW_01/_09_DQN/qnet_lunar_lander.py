# QNet for LunarLander-v3
# Adapted from c_qnet.py: changed default n_features=8, n_actions=4 to match LunarLander
#   - LunarLander state: 8-dim (x, y, vx, vy, angle, ang_vel, left_leg_contact, right_leg_contact)
#   - LunarLander action: 4 discrete (do nothing, fire left, fire main, fire right)
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, n_features: int = 8, n_actions: int = 4):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs: torch.Tensor, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            action = random.randrange(0, self.n_actions)
        else:
            q_values = self.forward(obs)
            action = torch.argmax(q_values, dim=-1)
            action = action.item()
        return action
