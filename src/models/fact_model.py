import torch.nn as nn
import torch.nn.functional as F


class FactDQN(nn.Module):

    def __init__(self, num_actions=5):

        super().__init__()

        self.fc1 = nn.Linear(34, 128)

        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 128)

        self.fc4 = nn.Linear(128, 32)

        self.fc5 = nn.Linear(32, num_actions)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = self.fc5(x)

        return x