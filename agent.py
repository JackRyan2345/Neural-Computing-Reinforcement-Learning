import copy
import random

import torch
import torch.nn as nn

from config import C
from model import QNetwork

class DQNAgent:
    def __init__(self, input_dim, output_dim, double = C.DOUBLE):
        self.n_actions = output_dim
        self.gamma = C.GAMMA
        self.tau = C.TAU
        self.double = double
        self.grad_clip = C.GRAD_CLIP

        #Two Networks
        self.qnetwork_local = QNetwork(input_dim, output_dim)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.qnetwork_target.eval()
        for p in self.qnetwork_target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=C.LR)
        self.loss = nn.SmoothL1Loss()


if __name__ == "__main__":
    agent = DQNAgent(6, 3)
    print(agent.qnetwork_local)                                   # should list fc1, fc2, fc3
    print(next(agent.qnetwork_target.parameters()).requires_grad)  # should print False
