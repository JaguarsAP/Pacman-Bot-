import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDQN(nn.Module):

    def __init__(self, num_actions=5):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        self.fc1 = nn.Linear(32 * 6 * 6, 32)

        self.fc2 = nn.Linear(32, num_actions)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x