import argparse
import csv
import random
import time

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.environment import AsterixEnvironment

from src.observations import (
    spatial_observation,
    fact_observation
)

from src.models.image_model import ImageDQN
from src.models.fact_model import FactDQN

from src.dqn import DQNAgent
from src.evaluate import evaluate_agent


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        choices=["spatial", "fact"],
        required=True
    )

    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=100000
    )

    parser.add_argument("--warmup", type=int, default=5000)

    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4
    )

    parser.add_argument(
        "--target-frequency",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=25000
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--log-frequency",
        type=int,
        default=5000
    )

    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument(
        "--device",
        default="cpu"
    )

    return parser.parse_args()


def get_epsilon(step, args):

    progress = min(
        step / max(args.epsilon_decay, 1),
        1.0
    )

    return (
        args.epsilon_start
        + progress
        * (args.epsilon_end - args.epsilon_start)
    )


def main():

    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.threads)

    device = torch.device(args.device)

    # Create environment
    env = AsterixEnvironment(
        seed=args.seed,
        render=False
    )

    num_actions = env.get_num_actions()

    # Select architecture and observation format
    if args.agent == "spatial":

        model = ImageDQN(num_actions)
        observation_function = spatial_observation

    else:

        model = FactDQN(num_actions)
        observation_function = fact_observation

    # Create DQN agent
    agent = DQNAgent(
        model=model,
        num_actions=num_actions,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        device=device,
        seed=args.seed
    )

    # Output directories
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    run_name = (
        f"{args.agent}_seed{args.seed}"
        f"_steps{args.steps}"
    )

    training_path = Path(
        f"results/{run_name}_training.csv"
    )

    evaluation_path = Path(
        f"results/{run_name}_evaluation.csv"
    )

    best_model_path = Path(
        f"checkpoints/{run_name}_best.pt"
    )

    final_model_path = Path(
        f"checkpoints/{run_name}_final.pt"
    )

    # Statistics
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_losses = deque(maxlen=1000)

    completed_episodes = 0
    best_eval_reward = float("-inf")

    state = observation_function(env.reset())

    start_time = time.perf_counter()

    print("\n=== TRAINING CONFIGURATION ===")

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    print("\n=== STARTING TRAINING ===")

    with (
        open(training_path, "w", newline="") as train_file,
        open(evaluation_path, "w", newline="") as eval_file
    ):

        train_writer = csv.writer(train_file)
        eval_writer = csv.writer(eval_file)

        train_writer.writerow([
            "step",
            "episode",
            "reward",
            "episode_length",
            "epsilon",
            "elapsed_seconds"
        ])

        eval_writer.writerow([
            "step",
            "mean_reward",
            "std_reward",
            "median_reward",
            "max_reward",
            "mean_length",
            "elapsed_seconds"
        ])

        # Main training loop
        for step in range(1, args.steps + 1):

            epsilon = get_epsilon(step - 1, args)

            # Force random exploration during warmup
            if step <= args.warmup:
                epsilon = 1.0

            # Select action
            action = agent.select_action(
                state,
                epsilon
            )

            # Interact with environment
            next_state, reward, terminated = env.step(
                action
            )

            next_state = observation_function(next_state)

            # Store experience
            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            # Optimize policy network
            if (
                step >= args.warmup
                and step % args.train_frequency == 0
            ):

                loss = agent.train_step()

                if loss is not None:
                    recent_losses.append(loss)

            # Synchronize target network
            if step % args.target_frequency == 0:

                agent.update_target_network()

            state = next_state

            # Handle episode termination
            if terminated:

                completed_episodes += 1

                recent_rewards.append(
                    env.episode_reward
                )

                recent_lengths.append(
                    env.episode_steps
                )

                train_writer.writerow([
                    step,
                    completed_episodes,
                    env.episode_reward,
                    env.episode_steps,
                    epsilon,
                    time.perf_counter() - start_time
                ])

                state = observation_function(env.reset())

            # Print training statistics
            if step % args.log_frequency == 0:

                mean_reward = (
                    np.mean(recent_rewards)
                    if recent_rewards else 0.0
                )

                mean_loss = (
                    np.mean(recent_losses)
                    if recent_losses else 0.0
                )

                elapsed = (
                    time.perf_counter() - start_time
                )

                print(
                    f"Step {step:,}/{args.steps:,} | "
                    f"Episodes: {completed_episodes} | "
                    f"Reward: {mean_reward:.3f} | "
                    f"Loss: {mean_loss:.5f} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Steps/sec: {step / elapsed:.1f}",
                    flush=True
                )

                train_file.flush()

            # Periodic evaluation
            if (
                step % args.eval_frequency == 0
                or step == args.steps
            ):

                results = evaluate_agent(
                    model=agent.policy_network,
                    observation_function=observation_function,
                    num_episodes=args.eval_episodes,
                    seed=10000,
                    device=device
                )

                elapsed = (
                    time.perf_counter() - start_time
                )

                eval_writer.writerow([
                    step,
                    results["mean_reward"],
                    results["std_reward"],
                    results["median_reward"],
                    results["max_reward"],
                    results["mean_length"],
                    elapsed
                ])

                eval_file.flush()

                print(
                    f"\n[EVALUATION] Step {step:,} | "
                    f"Mean reward: "
                    f"{results['mean_reward']:.3f} | "
                    f"Max reward: "
                    f"{results['max_reward']:.1f} | "
                    f"Mean length: "
                    f"{results['mean_length']:.1f}",
                    flush=True
                )

                # Save best-performing model
                if results["mean_reward"] > best_eval_reward:

                    best_eval_reward = results["mean_reward"]

                    torch.save({
                        "model_state_dict":
                            agent.policy_network.state_dict(),
                        "config": vars(args),
                        "step": step,
                        "evaluation": results
                    }, best_model_path)

                    print(
                        "New best model saved!",
                        flush=True
                    )

                print()

    # Save final model
    torch.save({
        "model_state_dict":
            agent.policy_network.state_dict(),
        "config": vars(args),
        "step": args.steps
    }, final_model_path)

    elapsed = time.perf_counter() - start_time

    print("\n=== TRAINING COMPLETE ===")

    print("Agent:", args.agent)
    print("Total steps:", args.steps)
    print("Completed episodes:", completed_episodes)
    print("Optimization steps:", agent.training_steps)
    print("Best evaluation reward:", best_eval_reward)
    print("Runtime (seconds):", round(elapsed, 2))
    print("Steps per second:", round(args.steps / elapsed, 2))

    print("Training log:", training_path)
    print("Evaluation log:", evaluation_path)
    print("Best model:", best_model_path)
    print("Final model:", final_model_path)


if __name__ == "__main__":
    main()