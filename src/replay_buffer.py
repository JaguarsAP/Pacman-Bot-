from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, capacity=100000, seed=42):

        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):

        transition = (
            state.copy(),
            int(action),
            float(reward),
            next_state.copy(),
            bool(terminated)
        )

        self.buffer.append(transition)

    def sample(self, batch_size, device="cpu"):

        batch = self.rng.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.as_tensor(
            np.stack(states),
            dtype=torch.float32,
            device=device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=device
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device
        )

        next_states = torch.as_tensor(
            np.stack(next_states),
            dtype=torch.float32,
            device=device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):

        return len(self.buffer)