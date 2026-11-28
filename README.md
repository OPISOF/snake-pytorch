# Snake Game

The Snake Game is powered with PyTorch. Snake has an AI agent. It learns to play and improves its effectiveness during multiple game sessions. 

The best DQN parameters were tuned with a genetic algorithm.  

## Installation

The program uses Python 3.11+

Install the required dependencies:

```bash
pip3 install -r requirements.txt
```
Setup is complete!

## Run types

There are some run types for specific purposes.

> After the first program run (any type) the program creates a folder `runs/` in the project. Inside `runs/` there are 3 folders for each run type: `GA/`, `manual_sweeps/`, `single_runs/`.  

### Single run

```bash
python3 snake.py
```
A single run is needed for demonstration. It uses default configuration. 

A single run uses a renderer for the visualisation. By default it's turned on at the start of the session.

A single run uses Matplotlib for plotting snake's results during the session. The plot is saved to the `runs/single_runs` folder.   

#### Controls

Control the game using the following keys:

- Space - Toggle rendering.
Disabling rendering significantly speeds up training. 

- Backspace to kill snake.

Enjoy yourself!

---

> Sweep experiments create timestamped folders (inside `GA/` or `manual_sweeps`). 

### Manual sweep

```bash
python3 sweep.py --manual-sweep
```

Manual sweeps are for experiments with your own parameters. 

`runs.toml` contains your own configuration.

Here is an example of the `runs.toml` structure:

```toml
seed = [0, 10, 42] # Seed values

[config_name]
# Here are written configuration parameters
inner_layer = 128 
eating_reward = 0.01
trainer_gamma = 0.95

```
Seed values are required. Each configuration will be tested with them.

> You can create as many configs as you want. 

> You don't have to write every parameter. Your configuration will use default values from config.toml for non-mentioned parameters. 

Inside the experiment folder will be saved each run's plot and a csv with the results of each run.  

### Genetic algorithm

```bash
python3 sweep.py --ga
```

The genetic algorithm powered by PyGad finds the best DQN parameters.  

The `ga.toml` file keeps some of the genetic algorithm parameters.You can change the number of generations (`num_generations`), the number of parents mating (`num_parents_mating`) and the solutions per population (`sol_per_pop`) there. 

> Other genetic algorithm parameters, such as selectable genes, `parent_selection_type`, `crossover_type` and `mutation_type` are currently set in the code.

After the experiments the results will be in the experiments folder. You can find there run plots, csv file, fitness_graph, which shows the algorithm's progress during the generations, and the best pygad solution (pygad_GA.pkl). 

## Config

> The `config.toml` contains the default configuration parameters. 

> The `runs.toml` contains configurations for manual sweeps.

### Parameters

- **`inner_layer`** is the size of hidden layer in the Pytorch model.
- **`trainer_gamma`** is discount factor controlling the importance of future rewards. 
- **`learning_rate`** is a key hyperparameter that controls how quickly a model learns by determining the step size during weight updates.
- **`exploration_games`** is a parameter that controls the number of exploration games.
- **`survival_reward`** is a reward for survival.
- **`death_reward`** is a reward for death.
- **`eating_reward`** is a reward for eating food.
- **`games_after_exploration`** is the number of games after the exploration. Without a value the run will last forever. Sweeps and the GA need it, so keep it in `config.toml` 

> A run lasts `exploration_games + games_after_exploration` games. 

- **`step_limit`**. A game ends when its total steps exceed `snake length * step_limit`. This parameter prevents the agent from being stuck (Sometimes during the exploration the snake circles and doesn't grow)
- **`seed`** is random seed value. Without a value a random seed will be created.

