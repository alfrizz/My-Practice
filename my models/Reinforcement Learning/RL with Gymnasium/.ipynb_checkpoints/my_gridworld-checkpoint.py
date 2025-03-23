import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MyGridWorld(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        """
        Initialize the GridWorld environment.
        - Define a 3x3 grid world (9 states: 0 to 8).
        - Define actions and observation space.
        - Setup transition probabilities and rendering mode.
        """
        super().__init__()
        self.grid_size = 3  # Define a 3x3 grid (total 9 states: 0 to 8)
        self.state = 0  # Start the agent at the initial state (state 0)
        self.action_space = spaces.Discrete(4)  # 4 possible actions: left, down, right, up
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)  # Total number of states (9)

        self.render_mode = render_mode  # Render mode ("rgb_array" for visual images, "human" for text-based)

        # Precompute transition probabilities for all states and actions
        self.P = self._create_transition_probabilities()

    
    def _create_transition_probabilities(self):
        """
        Define the transition dynamics for the grid world.
        - States are numbered from 0 (top-left) to 8 (bottom-right).
        - Movement follows grid boundaries.
        - Rewards:
          * -1 for regular steps.
          * -10 for penalty states (4 and 7).
          * 10 for reaching the goal state (state 8).
        """
        P = {}
        for s in range(self.grid_size * self.grid_size):  # Iterate over all states (0 to 8)
            P[s] = {a: [] for a in range(4)}  # Initialize actions (0: left, 1: down, 2: right, 3: up)

            # Calculate the current row and column from the state
            row, col = divmod(s, self.grid_size)

            for action in range(4):  # Define the results of each action
                next_row, next_col = row, col
                if action == 0 and col > 0:  # Move left
                    next_col -= 1
                elif action == 1 and row < self.grid_size - 1:  # Move down
                    next_row += 1
                elif action == 2 and col < self.grid_size - 1:  # Move right
                    next_col += 1
                elif action == 3 and row > 0:  # Move up
                    next_row -= 1

                next_state = next_row * self.grid_size + next_col
                reward = -1  # Default penalty for moving
                if next_state == 8:  # Goal state
                    reward = 10
                elif next_state in [4, 7]:  # Penalty states
                    reward = -10
                done = next_state == 8  # Episode ends upon reaching the goal state

                # Add the transition tuple (probability, next_state, reward, done)
                P[s][action].append((1.0, next_state, reward, done))
        return P

    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state (state 0).
        Returns:
        - Initial state (int).
        - Info dictionary (empty in this implementation).
        """
        self.state = 0  # Start at the top-left corner
        return self.state, {}  # Return the initial state and empty info dictionary

    
    def set_state(self, state):
        """
        Manually set the current state for debugging or testing.
        Parameters:
        - state (int): The new state to set.
        """
        self.state = state

    
    def step(self, action):
        """
        Execute an action and transition to the next state.
        Parameters:
        - action (int): The action to take (0 = left, 1 = down, 2 = right, 3 = up).
        Returns:
        - next_state (int): The updated state.
        - reward (float): The reward for the transition.
        - done (bool): True if the episode ends.
        - truncated (bool): Always False (not used here).
        - info (dict): Empty dictionary (no extra information).
        """
        # Get the transition details from the transition probability dictionary (deterministic)
        _, next_state, reward, done = self.P[self.state][action][0]

        # Update the current state
        self.state = next_state

        # Return the updated state, reward, termination flag, and additional info
        return self.state, reward, done, False, {}

        
    def render(self):
        """
        Render the grid visually with state numbers and markings:
        - "A" for the agent.
        - "X" for penalty states.
        - State numbers for all other states.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        for state in range(self.grid_size * self.grid_size):  # Fill grid with state numbers
            row, col = divmod(state, self.grid_size)
            grid[row, col] = str(state)  # Assign state number

        # Mark agent's current position with "A"
        row, col = divmod(self.state, self.grid_size)
        grid[row, col] = "A"

        # Mark penalty states with "X"
        for penalty_state in [4, 7]:
            penalty_row, penalty_col = divmod(penalty_state, self.grid_size)
            grid[penalty_row, penalty_col] = "X"

        # Render the grid (text-based for "human" mode)
        if self.render_mode == "human":
            print("\n".join([" ".join(row) for row in grid]))
        elif self.render_mode == "rgb_array":
            # Generate an RGB image with text and colors
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
        Perform cleanup (no cleanup needed for this simple implementation).
        """
        pass
