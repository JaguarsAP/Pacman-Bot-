import csv
import time
from pathlib import Path

import numpy as np

from src.environment import AsterixEnvironment


NUM_EPISODES = 1000
SEED = 42


def main():

    env = AsterixEnvironment(seed=SEED)
    rng = np.random.default_rng(SEED)

    rewards = []
    lengths = []

    start_time = time.perf_counter()

    for episode in range(NUM_EPISODES):

        env.reset()

        terminated = False

        while not terminated:

            action = rng.integers(env.get_num_actions())

            _, reward, terminated = env.step(action)

        rewards.append(env.episode_reward)
        lengths.append(env.episode_steps)

        if (episode + 1) % 100 == 0:

            print(
                f"Episode {episode + 1}/{NUM_EPISODES} | "
                f"Average reward: {np.mean(rewards):.3f}"
            )

    elapsed = time.perf_counter() - start_time

    print("\n=== RANDOM BASELINE RESULTS ===")

    print("Episodes:", NUM_EPISODES)
    print("Total steps:", env.total_steps)
    print("Mean reward:", np.mean(rewards))
    print("Reward std:", np.std(rewards))
    print("Median reward:", np.median(rewards))
    print("Max reward:", np.max(rewards))
    print("Mean episode length:", np.mean(lengths))
    print("Runtime:", round(elapsed, 2), "seconds")
    print("Steps per second:", env.total_steps / elapsed)

    # Save results
    Path("results").mkdir(exist_ok=True)

    with open(
        "results/random_baseline.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "reward",
            "episode_length"
        ])

        for i, (reward, length) in enumerate(
            zip(rewards, lengths)
        ):

            writer.writerow([i + 1, reward, length])

    print("\nResults saved to results/random_baseline.csv")


if __name__ == "__main__":
    main()