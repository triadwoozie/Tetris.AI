import sys
import time
import pygame

"""
usage: python3 run_model.py sample.keras (copy best.keras to src)

This script runs a pre-trained Deep Q-Network (DQN) model to play Tetris
with the same fall speed as the playable version.

Usage: python3 run_model.py <model_file>
(Example: python3 run_model.py sample.keras)

Key components and functionality:

1. Command-line Arguments:
   - The script expects a model file as a command-line argument.
   - It exits with an error message if no model file is provided.

2. Imports and Setup:
   - Imports the custom DQNAgent and Tetris environment.
   - Creates a Tetris environment instance.
   - Initializes Pygame for consistent timing.

3. Agent Initialization:
   - Initializes a DQNAgent with the provided model file.
   - The agent's state size is set based on the Tetris environment.

4. Game Loop:
   - Runs continuously until the game is over (done = True).
   - Uses a timing mechanism to control piece fall speed.
   - In each step:
     a. Gets all possible next states from the Tetris environment.
     b. The agent chooses the best state based on its trained model.
     c. Converts the chosen state back to an action (piece position and rotation).
     d. Plays the chosen action in the Tetris environment.
     e. Renders the game state (render=True).
   - Applies gravity to the piece at regular intervals based on the current level.

This script demonstrates how to use a trained DQN model to play Tetris autonomously
with a consistent fall speed. It's useful for visualizing the performance of a 
trained agent and can be used to evaluate the effectiveness of different training 
strategies or model architectures.
"""

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris import Tetris

# Initialize Pygame for consistent timing
pygame.init()

env = Tetris()
agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[1])
done = False

# Set up the clock for timing
clock = pygame.time.Clock()
fall_time = 0
fall_speed = 500  # Start with 0.5 seconds (500 ms) per gravity drop
fall_speed_by_level = {
    0: 500,
    1: 450,
    2: 400,
    3: 350,
    4: 300,
    5: 250,
    6: 200,
    7: 150,
    8: 100,
    9: 80,
    10: 50
}

while not done:
    # Get the time since last frame
    delta_time = clock.get_rawtime()
    clock.tick()

    # Increase fall time
    fall_time += delta_time

    # Update fall speed based on current level
    current_level = min(env.level, 10)  # Cap at level 10
    fall_speed = fall_speed_by_level[current_level]

    # Check if it's time for the piece to fall
    if fall_time >= fall_speed:
        fall_time = 0
        next_states = {tuple(v): k for k, v in env.get_next_states().items()}
        best_state = agent.best_state(next_states.keys())
        best_action = next_states[best_state]
        
        # Apply gravity
        _, done = env.play(best_action[0], best_action[1], render=True)

    # Handle Pygame events to keep the window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Render the game state
    env.render()

# Quit Pygame
pygame.quit()