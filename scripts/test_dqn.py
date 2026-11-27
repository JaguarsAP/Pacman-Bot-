import math

import torch

from src.environment import AsterixEnvironment

from src.observations import (
    spatial_observation,
    fact_observation
)

from src.models.image_model import ImageDQN
from src.models.fact_model import FactDQN

from src.dqn import DQNAgent


def test_agent(name, model, observation_function):

    print(f"\n=== TESTING {name.upper()} ===")

    env = AsterixEnvironment(seed=42)

    agent = DQNAgent(
        model=model,
        num_actions=env.get_num_actions(),
        batch_size=32,
        buffer_capacity=1000,
        seed=42
    )

    state = observation_function(env.reset())

    # Save initial network parameters
    initial_parameters = [
        parameter.detach().clone()
        for parameter in agent.policy_network.parameters()
    ]

    losses = []

    for step in range(200):

        action = agent.select_action(
            state,
            epsilon=1.0
        )

        assert 0 <= action < 5

        next_state, reward, terminated = env.step(action)

        next_state = observation_function(next_state)

        agent.store_transition(
            state,
            action,
            reward,
            next_state,
            terminated
        )

        loss = agent.train_step()

        if loss is not None:

            assert math.isfinite(loss)

            losses.append(loss)

        state = next_state

        if terminated:

            state = observation_function(env.reset())

    # Verify that optimization changed the parameters
    parameters_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            initial_parameters,
            agent.policy_network.parameters()
        )
    )

    assert parameters_changed

    # Test target network synchronization
    agent.update_target_network()

    for policy_param, target_param in zip(
        agent.policy_network.parameters(),
        agent.target_network.parameters()
    ):

        assert torch.equal(
            policy_param,
            target_param
        )

    print("Replay buffer size:", len(agent.replay_buffer))

    print("Optimization steps:", agent.training_steps)

    print("Initial loss:", losses[0])

    print("Final loss:", losses[-1])

    print("Parameters changed:", parameters_changed)

    print("Target network synchronized: True")

    print("ALL AGENT TESTS PASSED")


def main():

    torch.set_num_threads(1)

    torch.manual_seed(42)

    test_agent(
        "Spatial agent",
        ImageDQN(),
        spatial_observation
    )

    torch.manual_seed(42)

    test_agent(
        "Fact agent",
        FactDQN(),
        fact_observation
    )

    print("\nALL DQN TESTS PASSED")


if __name__ == "__main__":
    main()