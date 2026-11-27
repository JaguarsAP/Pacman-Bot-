from minatar import Environment


class AsterixEnvironment:

    def __init__(self, seed=42, render=False):

        self.env = Environment("asterix")
        self.env.seed(seed)

        self.render_enabled = render

        self.actions = self.env.minimal_action_set()
        self.num_actions = len(self.actions)

        self.total_steps = 0
        self.episode_steps = 0
        self.episode_reward = 0

    def reset(self):

        self.env.reset()

        self.episode_steps = 0
        self.episode_reward = 0

        return self.env.state()

    def step(self, action):

        # Convert network output index to MinAtar action
        game_action = self.actions[int(action)]

        # Execute action
        reward, terminated = self.env.act(game_action)

        # Update statistics
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += reward

        # Retrieve updated state
        state = self.env.state()

        if self.render_enabled:
            self.env.display_state(50)

        return state, reward, terminated

    def get_state(self):

        return self.env.state()

    def get_num_actions(self):

        return self.num_actions