from minatar import Environment
import numpy as np
RENDER = False

def main():

    # Initialize Asterix
    env = Environment("asterix")

    # Set random seeds
    env.seed(42)
    rng = np.random.default_rng(42)

    # Reset environment
    env.reset()

    # Get initial state
    state = env.state()

    # Get available actions
    actions = env.minimal_action_set()

    print("=== Environment Information ===")
    print("Observation shape:", state.shape)
    print("Observation type:", state.dtype)
    print("Available actions:", actions)
    print("Number of actions:", len(actions))
    print("Minimum value:", state.min())
    print("Maximum value:", state.max())

    episode = 1
    episode_reward = 0
    episode_steps = 0

    for step in range(1000):

        # Display the current game state
        if RENDER:
            env.display_state(50)

        # Select a random action
        action = int(rng.choice(actions))

        # Execute action
        reward, terminated = env.act(action)

        # Retrieve updated state
        next_state = env.state()

        episode_reward += reward
        episode_steps += 1

        state = next_state

        if terminated:

            print(
                f"Episode {episode} | "
                f"Reward: {episode_reward} | "
                f"Steps: {episode_steps}"
            )

            episode += 1
            episode_reward = 0
            episode_steps = 0

            env.reset()
            state = env.state()


if __name__ == "__main__":
    main()