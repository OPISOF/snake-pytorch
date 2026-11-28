import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import pygad
import tomllib as toml
from matplotlib import use as matplotlib_use

import snake

MAX_WORKERS = 4


PARAMETER_NAMES = [
    "inner_layer",
    "trainer_gamma",
    "learning_rate",
    "exploration_games",
    "survival_reward",
    "death_reward",
    "eating_reward",
    "step_limit",
]


class Config:
    def __init__(self):
        with open("config.toml", "rb") as file:
            self.default = toml.load(file)

    def build_sweep(self):
        with open("runs.toml", "rb") as file:
            self.sweep = toml.load(file)

        self.configs = []

        seeds = self.sweep["seed"]

        for seed in seeds:
            for config_name in self.sweep:
                if config_name != "seed":
                    experiment_config = self.sweep[config_name].copy()
                    experiment_config = self.merge_with_defaults(
                        experiment_config, seed
                    )
                    experiment_config["config_name"] = config_name
                    self.configs.append(experiment_config)

    def merge_with_defaults(self, experiment_config, seed=None):
        if seed is not None:
            experiment_config["seed"] = seed

        for key in self.default:
            if key not in experiment_config:
                experiment_config[key] = self.default[key]

        snake.get_limit_games(experiment_config)

        return experiment_config


class ResultsLog:
    def __init__(self, experiment_dir):

        self.filename = f"{experiment_dir}/stats.csv"
        path = Path(self.filename)

        if not path.exists():
            with open(self.filename, "a", newline="") as csvfile:
                fieldnames = (
                    ["name"]
                    + PARAMETER_NAMES
                    + ["max score", "mean_score (last 100 games)"]
                )

                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)

    def add(self, stats, file_name, config):
        parameters = [config[key] for key in PARAMETER_NAMES]

        with open(self.filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([file_name] + parameters + stats)


class GeneticAlgorithm:
    def __init__(self):
        with open("ga.toml", "rb") as file:
            self.ga_parameters = toml.load(file)

        self.gene_space = [
            [64, 128, 256, 512, 1024],  # inner_layer
            {"low": 0.9, "high": 1},  # trainer_gamma
            {"low": 10 ** (-5), "high": 10 ** (-3)},  # learning_rate
            {"low": 100, "high": 2000},  # exploration_games
            [0, 0.01],  # survival_reward
        ]

        self.gene_type = [int, float, float, int, float]

        self.num_genes = len(self.gene_space)

        self.parent_selection_type = "tournament"

        self.crossover_type = "uniform"

        self.mutation_type = "adaptive"
        self.mutation_num_genes = (2, 1)

        self.config = Config()
        self.run_paths = snake.RunPaths("GA")
        self.results = ResultsLog(self.run_paths.experiment_dir)

    def score_parameters(self, ga_instance, solution, solution_idx):
        parameters = {
            "inner_layer": solution[0],
            "trainer_gamma": round(solution[1], 3),
            "learning_rate": float(f"{solution[2]:.3g}"),
            "exploration_games": solution[3],
            "survival_reward": solution[4],
        }

        run_config = self.config.merge_with_defaults(parameters)
        stats = run_experiment(
            run_config,
            self.results,
            self.run_paths,
            solution_idx,
            generation=ga_instance.generations_completed,
        )
        fitness = stats[1]

        return fitness

    def on_generation(self, ga_instance):
        print(f"Generations completed: {ga_instance.generations_completed}")

    def run(self):
        ga_instance = pygad.GA(
            num_generations=self.ga_parameters["num_generations"],
            num_parents_mating=self.ga_parameters["num_parents_mating"],
            fitness_func=self.score_parameters,
            sol_per_pop=self.ga_parameters["sol_per_pop"],
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            gene_type=self.gene_type,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_num_genes=self.mutation_num_genes,
            parallel_processing=["process", MAX_WORKERS],
            on_generation=self.on_generation,
        )

        ga_instance.run()
        print("The process is finished! Look at the graph.")
        fitness_graph = ga_instance.plot_fitness()
        fitness_graph.savefig(f"{self.run_paths.experiment_dir}/fitness_graph.png")
        ga_instance.save(f"./{self.run_paths.experiment_dir}/pygad_GA")


def run_experiment(
    config, results, run_paths, run_number, generation=None, config_name=None
):

    randomizer = snake.set_seed(config)

    game = snake.SnakeGame(randomizer)
    renderer = None
    agent = snake.Agent(config, randomizer)
    matplotlib_use("Agg")
    plot = snake.Plot(enabled=False)

    snake.run(config, game, renderer, agent, plot)
    stats = [plot.max_score, plot.final_mean_score]

    if run_number is not None:
        run_paths.set_file_path_name(
            config["seed"],
            config_name=config_name,
            run_number=run_number,
            generation=generation,
        )

        plot.save(run_paths.file_path)

        results.add(stats, run_paths.file_name, config)

    return stats


def manual_main():
    jobs = []
    config = Config()
    config.build_sweep()
    processes = min(len(config.configs), MAX_WORKERS)
    run_paths = snake.RunPaths("manual_sweep")
    run_number = 0
    results = ResultsLog(run_paths.experiment_dir)

    with Pool(processes) as pool:
        for parameters in config.configs:
            run_number += 1
            jobs.append(
                pool.apply_async(
                    run_experiment,
                    args=(
                        parameters,
                        results,
                        run_paths,
                        run_number,
                    ),
                    kwds={"config_name": parameters["config_name"]},
                )
            )
        for job in jobs:
            job.get()


def main():
    genetic_algorithm = GeneticAlgorithm()
    genetic_algorithm.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-sweep", help="run manual sweep", action="store_true")
    parser.add_argument("--ga", help="run genetic algorithm", action="store_true")
    args = parser.parse_args()

    if args.manual_sweep or args.ga:
        start_time = time.time()

        if args.manual_sweep:
            manual_main()

        if args.ga:
            main()

        print("--- %s seconds ---" % (time.time() - start_time))

    else:
        parser.error("Choose --manual-sweep or --ga")
