# Snake Game

The Snake Game is powered with PyTorch. Snake has an AI agent. It learns to play and improves its effectiveness during multiple game sessions. 

The best DQN parameters were tuned with a genetic algorithm.  

## Installation

The program uses Python 3.11+

Install the required dependencies:

```bash
pip3 install -r requirements.txt
```
Then just start the program.

```bash
python3 snake.py
```

## Controls

Control the game using the following keys:

- Space - Toggle rendering.
Disabling rendering significantly speeds up training. 

- Backspace to kill snake.

Enjoy yourself!

## Config

> The `config.toml` contains the configuration parameters. 

### Parameters

- **`inner_layer`** is the size of hidden layer in the Pytorch model.
- **`trainer_gamma`** is discount factor controlling the importance of future rewards. 
- **`learning_rate`** is a key hyperparameter that controls how quickly a model learns by determining the step size during weight updates.
- **`exploration_games`** is a parameter that controls the number of exploration games.
- **`survival_reward`** is a reward for survival.
- **`death_reward`** is a reward for death.
- **`eating_reward`** is a reward for eating food.
- **`games_after_exploration`** is the number of games after the exploration. Without a value the run will last forever.
- **`seed`** is random seed value.
