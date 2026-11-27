import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        model,
        num_actions=5,
        learning_rate=0.0001,
        gamma=0.99,
        batch_size=32,
        buffer_capacity=100000,
        device="cpu",
        seed=42
    ):

        self.device = torch.device(device)

        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.rng = random.Random(seed)

        # Main network
        self.policy_network = model.to(self.device)

        # Independent target network
        self.target_network = copy.deepcopy(
            self.policy_network
        )

        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            seed=seed
        )

        self.training_steps = 0

    def select_action(self, state, epsilon=0.0):

        # Exploration
        if self.rng.random() < epsilon:

            return self.rng.randrange(self.num_actions)

        # Exploitation
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            terminated
        )

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.replay_buffer.sample(
            self.batch_size,
            self.device
        )

        # Current Q-value estimates
        current_q_values = self.policy_network(states)

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Compute Bellman targets
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(dim=1).values

            targets = (
                rewards
                + self.gamma
                * next_q_values
                * (1.0 - dones)
            )

        # Calculate loss
        loss = self.loss_function(
            current_q_values,
            targets
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        # Prevent excessively large gradients
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        return loss.item()

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )