import copy
import random

import torch
import torch.nn as nn

from config import C
from model import QNetwork

class DQNAgent:
    # input_dim is observation size, cos & sin + angular velocity
    # output_dim are the actions: left, nothing, right
    # Double is whether a double DQN is used or not
    def __init__(self, input_dim, output_dim, double = C.DOUBLE):
        self.n_actions = output_dim # needed for random actions
        self.gamma = C.GAMMA # discount factor: how much future reward counts
        self.tau = C.TAU # how fast the target network follows the online network
        self.double = double
        self.grad_clip = C.GRAD_CLIP # limit of gradient step, stops huge updates

        #Two Networks
        self.qnetwork_local = QNetwork(input_dim, output_dim) # online network: Gets trained
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local) # target network: a copy only to compute training targets
        self.qnetwork_target.eval()
        for p in self.qnetwork_target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=C.LR) # updates the online networks weights
        self.loss = nn.SmoothL1Loss() # Huber loss, squared error for small, linear for large

    def act(self, state, eps = 0.0):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return int(action_values.argmax(dim=1).item())


if __name__ == "__main__":
    agent = DQNAgent(6, 3)
    print(agent.qnetwork_local)                                   # should list fc1, fc2, fc3
    print(next(agent.qnetwork_target.parameters()).requires_grad)  # should print False
