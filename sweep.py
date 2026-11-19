import csv
import random
import time
from multiprocessing import Pool
from pathlib import Path

import pygad
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

    def update(self, sweep=None):
        if not sweep:
            sweep = dict(list(self.sweep.items())[1:]).copy()
        for seed in self.sweep["seed"]:
            for run_name in sweep:
                experiment_config = sweep.copy()
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


class GeneticAlgorithm:
    def __init__(self):
        self.num_generations = 50
        self.num_parents_mating = 4

        self.sol_per_pop = 8

        self.gene_space = [
            [64, 128, 256, 512, 1024],  # inner_layer
            {"low": 0.85, "high": 1},  # trainer_gamma
            {"low": 10 ** (-4), "high": 10 ** (-3)},  # learning_rate
            {"low": 100, "high": 2000},  # exploration_games
            [0, 0.01],  # survival_reward
        ]

        self.gene_type = [int, float, float, int, float]

        self.num_genes = len(self.gene_space)

        self.parent_selection_type = "sss"
        self.keep_parents = 1

        self.crossover_type = "single_point"

        self.mutation_type = "random"
        self.mutation_percent_genes = 10

        self.config = Config()
        self.experiment_dir = snake.make_experiment_dir()
        self.results = ResultsLog(self.experiment_dir)

    def score_parameters(self, ga_instance, solution, solution_idx):
        parameters = {
            "inner_layer": solution[0],
            "trainer_gamma": solution[1],
            "learning_rate": solution[2],
            "exploration_games": solution[3],
            "survival_reward": solution[4],
        }

        self.config.update(parameters)
        stats = run_config(self.config.configs[0], self.results, self.experiment_dir)
        fitness = stats[0] / (
            (snake.SCREEN_WIDTH * snake.SCREEN_HEIGHT) / snake.BLOCK**2
        )

        print("Solution checked!")
        return fitness

    def run(self):
        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.score_parameters,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            gene_type=self.gene_type,
            parent_selection_type=self.parent_selection_type,
            keep_parents=self.keep_parents,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_percent_genes=self.mutation_percent_genes,
            parallel_processing=["process", 4],
        )

        ga_instance.run()


def run_experiment(config, results, experiment_dir):
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
    return stats


def manual_main():
    jobs = []
    config = Config()
    config.update()
    processes = config.get_processes()
    experiment_dir = snake.make_experiment_dir()
    results = ResultsLog(experiment_dir)

    with Pool(processes) as pool:
        for parameters in config.configs:
            jobs.append(
                pool.apply_async(run_experiment, (parameters, results, experiment_dir))
            )
        for job in jobs:
            job.get()


def main():
    genetic_algorithm = GeneticAlgorithm()
    genetic_algorithm.run()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
