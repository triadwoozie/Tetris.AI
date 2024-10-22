# Tetris DQN

An implementation of a Deep Q-Network (DQN) agent that learns to play Tetris.

## Project Overview

This project uses reinforcement learning, specifically a Deep Q-Network (DQN), to train an AI agent to play Tetris. The agent learns to make decisions about piece placement and rotation to maximize its score in the game.

## Features

- Custom Tetris environment implementation
- DQN agent with configurable neural network architecture
- Training script with customizable hyperparameters
- Script to run and visualize trained models
- TensorBoard logging for training progress

## Requirements

- Python 3.7+
- PyTorch
- Pygame
- NumPy
- TensorFlow (for TensorBoard logging)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/triadwoozie/tetris-dqn.git
   cd tetris-dqn
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Agent

To train the DQN agent, run:

```
python train_dqn.py [--verbose]
```

- Use the `--verbose` flag to run in verbose mode (no rendering).

The script will save the best model as `best.keras` in the project directory.

### Running a Trained Model

To watch a trained model play Tetris:

1. Ensure you have a trained model file (e.g., `best.keras`) in the project directory.
2. Run:
   ```
   python run_model.py best.keras
   ```

## Project Structure

- `tetris.py`: Implementation of the Tetris game environment
- `dqn_agent.py`: DQN agent implementation
- `train_dqn.py`: Script for training the DQN agent
- `run_model.py`: Script for running a trained model
- `logs.py`: Custom TensorBoard logging utility

## Customization

You can customize various aspects of the training process by modifying the hyperparameters in `train_dqn.py`, such as:

- Number of episodes
- Neural network architecture
- Learning rate
- Replay memory size
- Epsilon decay rate

## Logging

Training progress is logged using TensorBoard. To view the logs, run:

```
tensorboard --logdir=logs
```

Then open your browser and go to `http://localhost:6006`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Pygame](https://www.pygame.org/) for the game rendering
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [OpenAI Gym](https://gym.openai.com/) for inspiration on environment design

## Contact

For any questions or feedback, please open an issue on this repository or contact [triadwoozie](https://github.com/triadwoozie).

# Tetris DQN

## Detailed Documentation

### Components Overview

#### 1. Tetris Environment (`tetris.py`)

The `Tetris` class in this file implements the Tetris game logic and provides an interface for the DQN agent to interact with.

Key methods:
- `reset()`: Initializes a new game state.
- `play(x, rotation, render=False, render_delay=None)`: Executes a move and returns the reward and game over status.
- `get_next_states()`: Returns all possible next states from the current state.
- `get_state_size()`: Returns the size of the state representation.

The environment uses a numerical representation of the game board and pieces, allowing for efficient state processing by the DQN agent.

#### 2. DQN Agent (`dqn_agent.py`)

The `DQNAgent` class implements the Deep Q-Network algorithm.

Key components:
- Neural Network: Implemented using PyTorch, with customizable architecture.
- Replay Memory: Stores experiences for off-policy learning.
- Epsilon-greedy Exploration: Balances exploration and exploitation during training.

Key methods:
- `best_state(states)`: Selects the best action based on Q-values.
- `add_to_memory(current, next, reward, done)`: Adds an experience to the replay memory.
- `train(batch_size, epochs)`: Performs a training step on a batch of experiences.

#### 3. Training Script (`train_dqn.py`)

This script orchestrates the training process of the DQN agent.

Key functionalities:
- Hyperparameter setup
- Training loop implementation
- Periodic evaluation and model saving
- Logging of training statistics

Customizable parameters include learning rate, batch size, memory size, and neural network architecture.

#### 4. Model Running Script (`run_model.py`)

This script loads a trained model and runs it in the Tetris environment.

Usage:
```
python run_model.py <model_file>
```

It visualizes the performance of the trained agent, allowing for evaluation and demonstration.

#### 5. Custom TensorBoard Logger (`logs.py`)

Implements a custom logging utility for TensorBoard, allowing detailed tracking of training progress.

### State Representation

The state of the Tetris game is represented as a vector with the following components:
1. Board state: A flattened representation of the game board.
2. Current piece: Encoded representation of the active Tetris piece.
3. Next piece: Encoded representation of the upcoming piece.

This compact representation allows the DQN to process the game state efficiently.

### Reward Structure

The reward function is designed to encourage the agent to clear lines and avoid game over:
- Positive reward for clearing lines (scaled by the number of lines cleared)
- Small negative reward for each step (encourages faster completion)
- Large negative reward for game over

### Training Process

1. Initialize the Tetris environment and DQN agent.
2. For each episode:
   a. Reset the environment to a new game state.
   b. While the game is not over:
      - Choose an action using epsilon-greedy strategy.
      - Execute the action in the environment.
      - Store the experience in the replay memory.
      - If enough experiences are collected, perform a training step.
   c. Log the episode statistics.
   d. If performance improves, save the model.

### Model Architecture

The DQN uses a feedforward neural network with the following default architecture:
- Input layer: Size matches the state representation
- Hidden layers: Configurable, default is [32, 32, 32]
- Output layer: Size matches the number of possible actions

Activation functions are ReLU for hidden layers and linear for the output layer.

### Hyperparameter Tuning

Key hyperparameters that can be tuned for better performance:
- Learning rate
- Epsilon decay rate
- Replay memory size
- Batch size
- Discount factor
- Neural network architecture

Experiment with these parameters to optimize the agent's performance for different Tetris configurations or learning objectives.

## Extending the Project

Possible extensions to this project:
1. Implement prioritized experience replay for more efficient learning.
2. Add support for different Tetris variants (e.g., different board sizes, piece sets).
3. Implement a multi-agent system where multiple DQN agents compete or cooperate.
4. Explore other reinforcement learning algorithms (e.g., Policy Gradient, A3C) for comparison.

For any questions about the implementation details or assistance with extending the project, please open an issue or contact the project maintainer.