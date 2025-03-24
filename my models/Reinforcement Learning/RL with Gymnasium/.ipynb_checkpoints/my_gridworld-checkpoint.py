import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MyGridWorld(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        """
        Initialize the GridWorld environment.
        - Define a 3x3 grid world with 9 states, numbered from 0 to 8.
        - Specify actions and observation space.
        - Set up the rendering mode.
        """
        super().__init__()
        self.grid_size = 3  # A 3x3 grid, resulting in 9 states (0 to 8)
        self.state = 0  # The agent starts at the initial state (state 0)
        self.action_space = spaces.Discrete(4)  # Four possible actions: left, down, right, up
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)  # Total states: 9

        self.render_mode = render_mode  # Render mode: "rgb_array" for visuals, "human" for text

        # Initialize P as a placeholder structure
        self.P = {s: {a: [] for a in range(4)} for s in range(self.grid_size * self.grid_size)}

        # Ensure the P variable is valid for the initial state
        self._initialize_p()

    def _initialize_p(self):
        """
        Ensure that every state-action pair in P has a valid default value.
        """
        for state in range(self.grid_size * self.grid_size):
            for action in range(4):  # Actions: 0 (left), 1 (down), 2 (right), 3 (up)
                self._populate_p_entry(state, action)

    def _populate_p_entry(self, state, action):
        """
        Populate the P[state][action] entry dynamically with the correct transition tuple.
        """
        row, col = divmod(state, self.grid_size)
        next_row, next_col = row, col

        # Check for wall collisions
        if action == 0 and col == 0:  # Hitting the left wall
            next_state = state
            reward = -2
        elif action == 1 and row == self.grid_size - 1:  # Hitting the bottom wall
            next_state = state
            reward = -2
        elif action == 2 and col == self.grid_size - 1:  # Hitting the right wall
            next_state = state
            reward = -2
        elif action == 3 and row == 0:  # Hitting the top wall
            next_state = state
            reward = -2
        else:
            # Update position if no wall collision occurs
            if action == 0:  # Left
                next_col -= 1
            elif action == 1:  # Down
                next_row += 1
            elif action == 2:  # Right
                next_col += 1
            elif action == 3:  # Up
                next_row -= 1

            next_state = next_row * self.grid_size + next_col
            reward = -1  # Default movement penalty
            if next_state == 8:  # Goal state
                reward = 10
            elif next_state in [4, 7]:  # Penalty states
                reward = -3

        done = next_state == 8
        self.P[state][action] = [(1.0, next_state, reward, done)]

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state (state 0).
        Returns:
        - Initial state (int).
        - Info dictionary (empty in this implementation).
        """
        self.state = 0  # Starting at the top-left corner
        return self.state, {}  # Returning the initial state and an empty info dictionary

    def set_state(self, state):
        """
        Set the current state manually for debugging or testing.
        Parameters:
        - state (int): The state to set as the current one.
        """
        self.state = state

    def step(self, action):
        """
        Perform an action and transition to the next state.
        Dynamically populate the P variable for the current state-action pair.
        Parameters:
        - action (int): Action to take (0 = left, 1 = down, 2 = right, 3 = up).
        Returns:
        - next_state (int): The updated state.
        - reward (float): Transition reward.
        - done (bool): True if the episode is finished.
        - truncated (bool): Always False (not used here).
        - info (dict): Extra information (empty dictionary here).
        """
        # Ensure P is populated for the current state and action
        if not self.P[self.state][action]:
            self._populate_p_entry(self.state, action)

        # Retrieve transition tuple
        prob, next_state, reward, done = self.P[self.state][action][0]

        # Update the current state
        self.state = next_state

        return self.state, reward, done, False, {}

    def render(self):
        """
        Render the grid visually with state numbers and indicators:
        - "A" marks the agent's position.
        - "X" represents penalty states.
        - Other states show their numbers.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        for state in range(self.grid_size * self.grid_size):  # Fill grid with state numbers
            row, col = divmod(state, self.grid_size)
            grid[row, col] = str(state)

        # Mark agent position with "A"
        row, col = divmod(self.state, self.grid_size)
        grid[row, col] = "A"

        # Mark penalty states with "X"
        for penalty_state in [4, 7]:
            penalty_row, penalty_col = divmod(penalty_state, self.grid_size)
            grid[penalty_row, penalty_col] = "X"

        # Render as text for "human" mode
        if self.render_mode == "human":
            print("\n".join([" ".join(row) for row in grid]))
        elif self.render_mode == "rgb_array":
            # Generate a visual grid with text and colors
            rgb_grid = np.ones((self.grid_size * 50, self.grid_size * 50, 3), dtype=np.uint8) * 255  # White background
            img = Image.fromarray(rgb_grid)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    state = r * self.grid_size + c
                    text = "A" if state == self.state else ("X" if state in [4, 7] else str(state))
                    color = (255, 0, 0) if text == "A" else (0, 0, 255) if text == "X" else (0, 0, 0)
                    draw.text((c * 50 + 20, r * 50 + 20), text, fill=color, font=font)
            return np.array(img)

    def close(self):
        """
        Perform cleanup tasks (not necessary here).
        """
        pass
