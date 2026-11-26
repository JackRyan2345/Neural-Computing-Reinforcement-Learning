import torch
import torch.nn as nn
from config import C

class QNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden = C.HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)