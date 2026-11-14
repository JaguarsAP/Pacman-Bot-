### py -3.13 main.py
import cv2
from collections import deque
import gymnasium as gym
import ale_py
import numpy as np

# Register Atari environments
gym.register_envs(ale_py)


def preprocess_frame(observation):

    # Converting our RGB image to grayscale
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    # Resize to 84 x 84 <-- need this for neural network Processing
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    return resized


###Frame Initialization :D   

class FrameStack:

    def __init__(self, num_frames=4):

        self.num_frames = num_frames

        # Store the most recent frames
        self.frames = deque(maxlen=num_frames)


    def reset(self, observation):

        # Preprocess initial observation
        frame = preprocess_frame(observation)

        # Initially duplicate the first frame
        for _ in range(self.num_frames):
            self.frames.append(frame)

        return self.get_state()


    def add_frame(self, observation):

        # Preprocess new observation
        frame = preprocess_frame(observation)

        # Automatically removes oldest frame when full
        self.frames.append(frame)

        return self.get_state()


    def get_state(self):

        # Combine frames into one state
        return np.stack(self.frames, axis=0)

def main():

    env = gym.make(
        "ALE/Pacman-v5",
        render_mode="human"
    )

    frame_stack = FrameStack(num_frames=4)

    observation, info = env.reset(seed=42)

    # Initialize our first state
    state = frame_stack.reset(observation)

    print("Initial state shape:", state.shape)
    print("Data type:", state.dtype)
    print("Minimum pixel:", state.min())
    print("Maximum pixel:", state.max())

    for step in range(1000):

        # Temporary random action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        # Process the new observation
        next_state = frame_stack.add_frame(observation)

        state = next_state

        if terminated or truncated:

            observation, info = env.reset()

            state = frame_stack.reset(observation)

    env.close()


if __name__ == "__main__":
    main()



