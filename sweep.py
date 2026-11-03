import random
import time
from multiprocessing import Pool

import tomllib as toml
import csv
import torch

import snake


class Config:
    def __init__(self):
        with open("runs.toml", "rb") as file:
            self.sweep = toml.load(file)
        with open("config.toml", "rb") as file:
            self.default = toml.load(file)
        self.configs = []

    def update(self):
        for seed in self.sweep["seed"]:
            for run_name in dict(list(self.sweep.items())[1:]):
                experiment_config = self.sweep[run_name].copy()
                experiment_config["seed"] = seed

                for key in self.default:
                    if key not in experiment_config:
                        experiment_config[key] = self.default[key]

                self.configs.append(experiment_config)

    def get_processes(self):
        if len(self.configs) >= 4:
            return 4
        else:
            return len(self.configs)


class ResultsLog:
    def __init__(self):
        pass        

def run_config(config, results, timestamp):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    game = snake.SnakeGame()
    renderer = None
    agent = snake.Agent(config)
    plot = snake.Plot(timestamp)
    plot.enabled = False

    stats = snake.run(config, game, renderer, agent, plot)  


def main():
    jobs = []
    config = Config()
    config.update()
    processes = config.get_processes()
    results = ResultsLog()
    timestamp = snake.make_experiment_dir()

    with Pool(processes) as pool:
        for parameters in config.configs:
            jobs.append(pool.apply_async(run_config, (parameters, results, timestamp)))
        for job in jobs:
            job.get()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
