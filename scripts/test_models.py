import torch
import numpy as np

from src.environment import AsterixEnvironment

from src.observations import (
    spatial_observation,
    fact_observation
)

from src.models.image_model import ImageDQN
from src.models.fact_model import FactDQN


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    torch.manual_seed(42)

    env = AsterixEnvironment(seed=42)

    state = env.reset()

    num_actions = env.get_num_actions()

    image_model = ImageDQN(num_actions)
    fact_model = FactDQN(num_actions)

    # Generate observations
    spatial = spatial_observation(state)
    facts = fact_observation(state)

    # Convert to PyTorch tensors
    spatial_tensor = torch.from_numpy(spatial)
    fact_tensor = torch.from_numpy(facts)

    # Add batch dimension
    spatial_tensor = spatial_tensor.unsqueeze(0)
    fact_tensor = fact_tensor.unsqueeze(0)

    print("=== SPATIAL MODEL ===")

    print(image_model)

    print("Input shape:", spatial_tensor.shape)

    print("Parameters:", count_parameters(image_model))

    with torch.no_grad():

        image_output = image_model(spatial_tensor)

    print("Output shape:", image_output.shape)
    print("Q-values:", image_output)

    print("Selected action:", image_output.argmax(dim=1).item())

    print("\n=== FACT MODEL ===")

    print(fact_model)

    print("Input shape:", fact_tensor.shape)

    print("Parameters:", count_parameters(fact_model))

    with torch.no_grad():

        fact_output = fact_model(fact_tensor)

    print("Output shape:", fact_output.shape)
    print("Q-values:", fact_output)

    print("Selected action:", fact_output.argmax(dim=1).item())

    # Test single observations
    assert image_output.shape == (1, 5)
    assert fact_output.shape == (1, 5)

    assert torch.isfinite(image_output).all()
    assert torch.isfinite(fact_output).all()

    # Test batched observations
    batch_size = 32

    spatial_batch = spatial_tensor.repeat(
        batch_size, 1, 1, 1
    )

    fact_batch = fact_tensor.repeat(
        batch_size, 1
    )

    image_batch_output = image_model(spatial_batch)
    fact_batch_output = fact_model(fact_batch)

    print("\n=== BATCH TEST ===")

    print("Spatial batch:", spatial_batch.shape)
    print("Spatial output:", image_batch_output.shape)

    print("Fact batch:", fact_batch.shape)
    print("Fact output:", fact_batch_output.shape)

    assert image_batch_output.shape == (32, 5)
    assert fact_batch_output.shape == (32, 5)

    # Test backpropagation
    image_model.zero_grad()
    fact_model.zero_grad()

    image_batch_output.mean().backward()
    fact_batch_output.mean().backward()

    for model in [image_model, fact_model]:

        for parameter in model.parameters():

            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    print("\nBACKPROPAGATION TEST PASSED")

    print("ALL MODEL TESTS PASSED")


if __name__ == "__main__":
    main()