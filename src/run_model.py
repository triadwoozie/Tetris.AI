import sys
"""
usage: python3 run_model.py sample.keras (copy best.keras to src)

This script runs a pre-trained Deep Q-Network (DQN) model to play Tetris. 

Usage: python3 run_model.py <model_file>
(Example: python3 run_model.py sample.keras)

Key components and functionality:

1. Command-line Arguments:
   - The script expects a model file as a command-line argument.
   - It exits with an error message if no model file is provided.

2. Imports and Setup:
   - Imports the custom DQNAgent and Tetris environment.
   - Creates a Tetris environment instance.

3. Agent Initialization:
   - Initializes a DQNAgent with the provided model file.
   - The agent's state size is set based on the Tetris environment.

4. Game Loop:
   - Runs continuously until the game is over (done = True).
   - In each step:
     a. Gets all possible next states from the Tetris environment.
     b. The agent chooses the best state based on its trained model.
     c. Converts the chosen state back to an action (piece position and rotation).
     d. Plays the chosen action in the Tetris environment.
     e. Renders the game state (render=True).

This script demonstrates how to use a trained DQN model to play Tetris autonomously. 
It's useful for visualizing the performance of a trained agent and can be used to 
evaluate the effectiveness of different training strategies or model architectures.
"""

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris import Tetris

env = Tetris()
agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[1])
done = False

while not done:
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1], render=True)
