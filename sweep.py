import random
import time
from multiprocessing import Pool

import tomllib as toml
import torch

import snake


class Config:
    def __init__(self):
        with open("runs.toml", "rb") as file:
            self.sweep = toml.load(file)
        with open("config.toml", "rb") as file:
            self.default = toml.load(file)
        self.list = []

    def update(self):
        for current_config in self.sweep:
            for key in self.default:
                if key not in current_config:
                    self.sweep[current_config][key] = self.default[key]

            self.list.append(self.sweep[current_config])


def run_config(config):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    game = snake.SnakeGame()
    renderer = None
    agent = snake.Agent(config)
    plot = snake.Plot()

    snake.run(config, game, renderer, agent, plot)


def main():
    process_arr = []
    processes = 4
    config = Config()
    config.update()

    with Pool(processes) as pool:
        for parameters in config.list:
            process_arr.append(pool.apply_async(run_config, (parameters,)))
        for i in process_arr:
            i.get()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
