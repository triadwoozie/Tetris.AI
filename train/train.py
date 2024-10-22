from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
import argparse
import os
import shutil
"""
This Script uses a dumbed down version of tetris.py for faster performance.

# DQN Tetris Agent Training

This script runs a Deep Q-Network (DQN) agent to play Tetris using a custom environment. 

## Overview

- **Environment**: The `Tetris` class is the game environment where the agent will play.
- **Agent**: The agent is implemented using the `DQNAgent` class, which utilizes a neural network to make decisions based on the current game state.
- **Logging**: The `CustomTensorBoard` class is used to log the training statistics for monitoring purposes.

## Parameters

### Training Parameters

- **total_episodes**: `int` - Total number of episodes for training (default: 3000).
- **max_steps**: `int` or `None` - Maximum number of steps per game (default: `None`, which allows infinite steps).
- **epsilon_stop_episode**: `int` - Episode at which random exploration stops (default: 2000).
- **memory_size**: `int` - Maximum number of steps stored in the agent's memory (default: 1000).
- **discount_factor**: `float` - Discount factor for the Q-learning formula (default: 0.95).
- **batch_size**: `int` - Number of actions to consider in each training batch (default: 128).
- **train_every**: `int` - Train the agent every `x` episodes (default: 1).
- **render_every**: `int` - Render gameplay every `x` episodes (default: 50).
- **render_delay**: `int` or `None` - Delay added to render each frame (default: `None`).
- **log_every**: `int` - Log current stats every `x` episodes (default: 50).
- **replay_start_size**: `int` - Minimum steps in memory required to start training (default: 1000).
- **neurons**: `list of int` - Number of neurons in each layer of the neural network (default: `[32, 32, 32]`).
- **activations**: `list of str` - Activation functions for the neural network layers (default: `['relu', 'relu', 'relu', 'linear']`).
- **save_best_model**: `bool` - Flag to save the best model so far at "best.keras" (default: `True`).

## Agent Initialization

The agent is initialized with the following parameters:

- State size from `env.get_state_size()`
- Neural network architecture and activation functions
- Epsilon stopping episode, memory size, discount factor, and replay start size

## Logging

Logs are created using the `CustomTensorBoard` class, with the directory structured as follows:

```
logs/tetris-nn=[neurons]-mem=[memory_size]-bs=[batch_size]-e=[train_every]-[timestamp]
```

## Main Training Loop

1. **Reset the environment**: At the start of each episode, reset the game.
2. **Game Loop**: For each step:
   - Determine the best action using the current state.
   - Play the action in the environment and receive a reward and done flag.
   - Store the transition in the agent's memory.
3. **Training**: Train the agent every `train_every` episodes using a batch of actions.
4. **Logging**: Log the average, minimum, and maximum scores every `log_every` episodes.
5. **Model Saving**: Save the model if the current score is the best so far.

## Running the Script

To execute the script, ensure that all dependencies are installed, then run the script directly. The agent will start training in the Tetris environment.

### Example:

python dqn_tetris.py

The best model will be saved as "best.keras" based on the highest score achieved during training.
"""
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Tetris.")
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode (no rendering).')
    return parser.parse_args()

def dqn(verbose):
    env = Tetris()
    
    # Training parameters
    total_episodes = 3000  # Total number of episodes
    max_steps = None       # Max number of steps per game (None for infinite)
    epsilon_stop_episode = 2000  # Episode at which random exploration stops
    memory_size = 1000     # Maximum number of steps stored by the agent
    discount_factor = 0.95  # Discount factor in the Q-learning formula
    batch_size = 128        # Number of actions to consider in each training
    train_every = 1         # Train every x episodes
    render_every = 50       # Render gameplay every x episodes
    render_delay = None     # Delay added to render each frame (None for no delay)
    log_every = 50          # Log current stats every x episodes
    replay_start_size = 1000  # Minimum steps in memory to start training
    neurons = [32, 32, 32]   # Number of neurons for each activation layer
    activations = ['relu', 'relu', 'relu', 'linear']  # Activation functions
    save_best_model = True    # Save the best model so far at "best.keras"

    # Initialize the agent
    agent = DQNAgent(
        env.get_state_size(),
        n_neurons=neurons,
        activations=activations,
        epsilon_stop_episode=epsilon_stop_episode,
        mem_size=memory_size,
        discount=discount_factor,
        replay_start_size=replay_start_size
    )

    # Set up logging
    log_dir = f'logs/tetris-nn={str(neurons)}-mem={memory_size}-bs={batch_size}-e={train_every}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    best_score = 0

    # Main training loop
    for episode in tqdm(range(total_episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        # Determine if rendering is needed
        render = not verbose and (episode % render_every == 0 if render_every else False)

        # Game loop
        while not done and (not max_steps or steps < max_steps):
            # State -> Action
            next_states = {tuple(v): k for k, v in env.get_next_states().items()}
            best_state = agent.best_state(next_states.keys())
            best_action = next_states[best_state]

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)

            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            steps += 1

        scores.append(env.get_game_score())

        # Train the agent
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=train_every)

        # Logging
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)

        # Save the model if it's the best score so far
        if save_best_model and env.get_game_score() > best_score:
            print(f'Saving a new best model (score={env.get_game_score()}, episode={episode})')
            best_score = env.get_game_score()
            agent.save_model("best.keras")
            # Code to copy "best.keras" to "src" directory
            if save_best_model:
                import shutil  # Import shutil for file operations
            # Check if "src" directory exists
            if os.path.exists("src"):
                # Copy "best.keras" to "src" directory
                shutil.copy("best.keras", os.path.join("src", "best.keras"))
                print(f'Copied best model to src/best.keras')
            else:
                print(f'src directory not found. Could not copy best.keras')        

        


if __name__ == "__main__":
    args = parse_arguments()
    dqn(args.verbose)
