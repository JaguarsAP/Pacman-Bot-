import numpy as np


def spatial_observation(state):

    return state.transpose(2, 0, 1).astype(np.float32)


def fact_observation(state):

    features = []

    # Extract player coordinates
    player_y, player_x = np.argwhere(state[:, :, 0])[0]

    # Normalize coordinates
    features.extend([
        player_x / 9.0,
        player_y / 9.0
    ])

    # Iterate through the eight playable rows
    for y in range(1, 9):

        enemies = np.flatnonzero(state[y, :, 1])
        trails = np.flatnonzero(state[y, :, 2])
        gold = np.flatnonzero(state[y, :, 3])

        # Determine entity type
        if len(enemies) > 0:

            x = int(enemies[0])
            is_gold = 0.0

        elif len(gold) > 0:

            x = int(gold[0])
            is_gold = 1.0

        else:

            # No entity in this row
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue

        # Determine movement direction
        direction = 0.0

        if len(trails) > 0:

            trail_x = int(trails[0])

            if trail_x < x:
                direction = 1.0

            elif trail_x > x:
                direction = -1.0

        # Add entity features
        features.extend([
            1.0,
            x / 9.0,
            direction,
            is_gold
        ])

    return np.array(features, dtype=np.float32)