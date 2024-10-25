### DQNAgent Class Documentation

The `DQNAgent` class implements a Deep Q-Learning agent using the Maximin approach. This agent is designed to learn optimal strategies in environments by estimating the expected score for different states.

The **Maximin approach** and **Q-learning** are concepts from decision-making and reinforcement learning, respectively. Here’s a breakdown of each:

### Maximin Approach

The Maximin approach is a decision-making strategy used in situations where a decision-maker seeks to minimize potential losses while maximizing the minimum gain. It is commonly applied in scenarios characterized by uncertainty or where the worst-case scenario is a primary concern. The basic idea is as follows:

1. **Identify Outcomes**: Consider all possible actions and their outcomes.
2. **Evaluate Minimum Gains**: For each action, determine the worst possible outcome (minimum gain).
3. **Maximize the Minimum**: Choose the action that has the highest minimum gain. In other words, select the option that offers the best worst-case scenario.

**Example**: In a game where a player wants to avoid losing, they may choose a strategy that ensures they lose the least amount possible, even if it means not winning the most.

### Q-Learning

Q-learning is a popular reinforcement learning algorithm used to learn the value of actions in a given state. It allows an agent to learn how to optimally act in an environment to maximize cumulative rewards. The core concepts of Q-learning include:

2. **Q-Values**: The Q-value, or action-value function \( Q(s, a) \), represents the expected future rewards when taking action \( a \) in state \( s \) and following the optimal policy thereafter.

3. **Exploration vs. Exploitation**: The agent must balance exploring new actions (to discover better rewards) and exploiting known actions that yield high rewards. This is often managed using strategies like ε-greedy, where the agent chooses a random action with probability ε and the best-known action with probability \(1 - \epsilon\).

4. **Convergence**: With sufficient exploration and under certain conditions, Q-learning can converge to the optimal action-value function, allowing the agent to derive the optimal policy.

### Relationship Between the Two

The Maximin approach can be seen as a conservative strategy that focuses on ensuring a reasonable outcome in uncertain conditions, while Q-learning is a method for learning the best actions over time through experience and rewards. In a reinforcement learning context, one might use a Maximin approach to guide the reward structure or to select actions that safeguard against the worst-case scenarios, influencing the learning process.
#### Class Definition

```python
class DQNAgent:
    '''
    Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): Importance of future rewards [0,1]
        epsilon (float): Exploration probability at the start
        epsilon_min (float): Minimum exploration probability
        epsilon_stop_episode (int): Episode to stop reducing exploration
        n_neurons (list(int)): Neurons in each inner layer
        activations (list): Activations for inner layers and output
        loss (obj): Loss function
        optimizer (obj): Optimizer used
        replay_start_size: Minimum size needed to train
        modelFile: Previously trained model file path
    '''
```

#### Initialization

The constructor initializes the agent's parameters and constructs the neural network model.

```python
def __init__(self, state_size, mem_size=10000, discount=0.95,
             epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
             n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
             loss='mse', optimizer='adam', replay_start_size=None, modelFile=None):
    ...
```

#### Build Model

This method builds the neural network model using Keras.

```python
def _build_model(self):
    '''Builds a Keras deep neural network model'''
    model = Sequential()
    model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

    for i in range(1, len(self.n_neurons)):
        model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

    model.add(Dense(1, activation=self.activations[-1]))
    model.compile(loss=self.loss, optimizer=self.optimizer)
    
    return model
```

#### Memory Management

The agent can store experiences in a replay buffer.

```python
def add_to_memory(self, current_state, next_state, reward, done):
    '''Adds a play to the replay memory buffer'''
    self.memory.append((current_state, next_state, reward, done))
```

#### Random Value Prediction

The agent can generate random values for exploration.

```python
def random_value(self):
    '''Random score for a certain action'''
    return random.random()
```

#### Value Prediction

The agent predicts the score for a given state.

```python
def predict_value(self, state):
    '''Predicts the score for a certain state'''
    return self.model.predict(state, verbose=0)[0]
```

#### Action Selection

The agent decides to act based on the current state and its exploration strategy.

```python
def act(self, state):
    '''Returns the expected score of a certain state'''
    state = np.reshape(state, [1, self.state_size])
    if random.random() <= self.epsilon:
        return self.random_value()
    else:
        return self.predict_value(state)
```

#### Best State Selection

The agent can evaluate a set of states to find the best one.

```python
def best_state(self, states):
    '''Returns the best state for a given collection of states'''
    ...
```

#### Training the Agent

The agent is trained using experiences stored in memory.

```python
def train(self, batch_size=32, epochs=3):
    '''Trains the agent'''
    ...
```

#### Save Model

This method allows saving the trained model to a file.

```python
def save_model(self, name):
    '''Saves the current model.
    It is recommended to name the file with the ".keras" extension.'''
    self.model.save(name)
```

### Example Usage

Here’s how you might create and use the `DQNAgent`:

```python
# Create the DQN agent
agent = DQNAgent(state_size=10, mem_size=10000, discount=0.95)

# Adding experiences to memory
agent.add_to_memory(current_state, next_state, reward, done)

# Train the agent
agent.train(batch_size=32, epochs=3)

# Save the trained model
agent.save_model("dqn_agent.keras")
```
