import csv
import random
import time
from multiprocessing import Pool
from pathlib import Path

import tomllib as toml
import torch
from matplotlib import use as matplotlib_use

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
    def __init__(self, experiment_dir):
        self.filename = f"{experiment_dir}/stats.csv"
        path = Path(self.filename)
        if not path.exists():
            with open(self.filename, "a", newline="") as csvfile:
                fieldnames = ["name", "max score", "mean_score (last 100 games)"]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)

    def add(self, stats, config):
        run_name = snake.get_run_name(config)
        with open(self.filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([run_name] + stats)


def run_config(config, results, experiment_dir):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    game = snake.SnakeGame()
    renderer = None
    agent = snake.Agent(config)
    matplotlib_use("Agg")
    plot = snake.Plot(experiment_dir)
    plot.enabled = False

    stats = snake.run(config, game, renderer, agent, plot)
    results.add(stats, config)


def main():
    jobs = []
    config = Config()
    config.update()
    processes = config.get_processes()
    experiment_dir = snake.make_experiment_dir()
    results = ResultsLog(experiment_dir)

    with Pool(processes) as pool:
        for parameters in config.configs:
            jobs.append(
                pool.apply_async(run_config, (parameters, results, experiment_dir))
            )
        for job in jobs:
            job.get()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
