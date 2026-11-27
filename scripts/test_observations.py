import numpy as np

from src.environment import AsterixEnvironment
from src.observations import (
    spatial_observation,
    fact_observation
)


def main():

    env = AsterixEnvironment(seed=42)
    rng = np.random.default_rng(42)

    state = env.reset()

    spatial = spatial_observation(state)
    facts = fact_observation(state)

    print("=== Initial Observation ===")

    print("Original shape:", state.shape)

    print("Spatial shape:", spatial.shape)
    print("Spatial dtype:", spatial.dtype)

    print("Fact shape:", facts.shape)
    print("Fact dtype:", facts.dtype)

    print("Player coordinates:", facts[:2])

    print("Entity features:")
    print(facts[2:].reshape(8, 4))

    # Verify dimensions
    assert spatial.shape == (4, 10, 10)
    assert facts.shape == (34,)

    print("\n=== Random Gameplay Test ===")

    for step in range(1000):

        action = rng.integers(env.get_num_actions())

        state, reward, terminated = env.step(action)

        spatial = spatial_observation(state)
        facts = fact_observation(state)

        assert spatial.shape == (4, 10, 10)
        assert facts.shape == (34,)

        assert np.isfinite(spatial).all()
        assert np.isfinite(facts).all()

        # Print a few example states
        if step in [20, 50, 100]:

            print(f"\nStep {step}")
            print("Player:", facts[:2])
            print("Entities:")
            print(facts[2:].reshape(8, 4))

        if terminated:
            state = env.reset()

    print("\nALL OBSERVATION TESTS PASSED")


if __name__ == "__main__":
    main()