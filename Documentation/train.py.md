### Imports and Documentation

```python
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
import argparse
import os
import shutil
```

This section imports the necessary modules for the script. `DQNAgent` and `Tetris` handle the agent's implementation and the game environment, respectively. `datetime` and `statistics` provide functionalities for logging and score calculations. `CustomTensorBoard` is used for monitoring training, while `tqdm` helps create progress bars. The `argparse`, `os`, and `shutil` modules are for command-line argument parsing and file management.

```python
"""
This Script uses a dumbed down version of tetris.py for faster performance.

# DQN Tetris Agent Training

This script runs a Deep Q-Network (DQN) agent to play Tetris using a custom environment. 

## Overview
...
"""
```

This is a multi-line comment that provides an overview of the script, detailing its purpose, structure, parameters, agent initialization, logging, the main training loop, and instructions for running the script.

### Argument Parsing Function

```python
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Tetris.")
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode (no rendering).')
    return parser.parse_args()
```

This function sets up argument parsing for the script. It allows the user to run the script in verbose mode by including the `--verbose` flag, which disables rendering of the game during training.

### DQN Training Function

```python
def dqn(verbose):
    env = Tetris()
```

The `dqn` function begins by creating an instance of the `Tetris` environment, setting up everything needed for training the DQN agent.

### Training Parameters

```python
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
```

This section defines various training parameters, including the total number of training episodes, maximum steps per game, exploration parameters, memory size, discount factor for Q-learning, batch size for training, logging intervals, and the neural network architecture for the DQN agent.

### Agent Initialization

```python
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
```

This code initializes the DQN agent with parameters defined earlier, including the state size from the environment, neural network architecture, exploration settings, and memory size.

### Logging Setup

```python
    # Set up logging
    log_dir = f'logs/tetris-nn={str(neurons)}-mem={memory_size}-bs={batch_size}-e={train_every}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)
```

This section sets up logging using `CustomTensorBoard`, creating a log directory with a structured naming convention that includes the neural network configuration and the current timestamp.

### Main Training Loop

```python
    scores = []
    best_score = 0

    # Main training loop
    for episode in tqdm(range(total_episodes)):
        current_state = env.reset()
        done = False
        steps = 0
```

Here, a list to store scores and a variable for the best score are initialized. The main training loop starts, iterating over the total number of episodes. The environment is reset at the beginning of each episode, and flags for completion and step count are initialized.

### Rendering Control

```python
        # Determine if rendering is needed
        render = not verbose and (episode % render_every == 0 if render_every else False)
```

This line determines whether to render the game visuals based on the verbose setting and the current episode number.

### Game Loop

```python
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
```

This inner loop runs the game until the episode ends or the maximum steps are reached. It retrieves the possible next states and determines the best action using the agent's policy. The action is executed in the environment, and the agent’s memory is updated with the transition information.

### Score Tracking

```python
        scores.append(env.get_game_score())
```

After each episode, the score achieved in the game is recorded.

### Agent Training

```python
        # Train the agent
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=train_every)
```

If the current episode number is a multiple of `train_every`, the agent is trained using a batch of actions.

### Logging Scores

```python
        # Logging
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)
```

Scores are logged at regular intervals, calculating the average, minimum, and maximum scores from the last logged episodes.

### Model Saving

```python
        # Save the model if it's the best score so far
        if save_best_model and env.get_game_score() > best_score:
            print(f'Saving a new best model (score={env.get_game_score()}, episode={episode})')
            best_score = env.get_game_score()
            agent.save_model("best.keras")
```

If the current score exceeds the previous best score, the model is saved, and the best score variable is updated.

### Copying Best Model

```python
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
```

This part attempts to copy the saved model to a "src" directory if it exists. It includes print statements to indicate success or failure in copying the model.

### Main Execution Block

```python
if __name__ == "__main__":
    args = parse_arguments()
    dqn(args.verbose)
```

This block runs the script, parsing arguments and calling the `dqn` function with the parsed verbosity setting.

---
