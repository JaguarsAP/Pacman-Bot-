import numpy as np
import torch

from src.environment import AsterixEnvironment


def evaluate_agent(
    model,
    observation_function,
    num_episodes=100,
    seed=10000,
    device="cpu",
    max_episode_steps=10000
):

    rewards = []
    lengths = []

    was_training = model.training
    model.eval()

    with torch.no_grad():

        for episode in range(num_episodes):

            env = AsterixEnvironment(
                seed=seed + episode,
                render=False
            )

            state = observation_function(env.reset())

            terminated = False

            while (
                not terminated
                and env.episode_steps < max_episode_steps
            ):

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                q_values = model(state_tensor)

                action = q_values.argmax(dim=1).item()

                next_state, reward, terminated = env.step(
                    action
                )

                state = observation_function(next_state)

            rewards.append(env.episode_reward)
            lengths.append(env.episode_steps)

    model.train(was_training)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths))
    }