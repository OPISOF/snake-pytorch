import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pygame
import tomllib
import torch
from torch import nn, optim

SCREEN_WIDTH, SCREEN_HEIGHT = 720, 480
BLOCK = 20
FPS = 40
PLOT_RATE = 200
PLOT_MAX_SCORE = 150

COMPASS = ["UP", "RIGHT", "DOWN", "LEFT"]


class SnakeGame:
    def __init__(self, randomizer):
        self.count = 0
        self.randomizer = randomizer
        self.reset()

    def reset(self):
        self.snake = [[100, 100], [100 + BLOCK, 100]]
        self.direction = "DOWN"
        self.food = spawn_food(self.snake, self.randomizer)
        self.score = 0
        self.count += 1
        self.total_steps = 0

    def step(self, action, config):
        reward = config["survival_reward"]

        self.decode_turn(action)
        head_x, head_y = self.snake[0]

        if self.direction == "DOWN":
            head_y += BLOCK
        elif self.direction == "UP":
            head_y -= BLOCK
        elif self.direction == "RIGHT":
            head_x += BLOCK
        elif self.direction == "LEFT":
            head_x -= BLOCK

        self.total_steps += 1

        if (
            is_crash(head_x, head_y, self.snake)
            or self.total_steps > len(self.snake) * config["step_limit"]
        ):
            reward = config["death_reward"]
            return True, reward

        self.snake.insert(0, [head_x, head_y])

        if self.snake[0] != self.food:
            self.snake.pop()
        else:
            self.score += 1
            reward = config["eating_reward"]

            if len(self.snake) == (SCREEN_WIDTH * SCREEN_HEIGHT) // BLOCK**2:
                return True, reward

            self.food = spawn_food(self.snake, self.randomizer)

        return False, reward

    def get_state(self):
        head_x, head_y = self.snake[0]

        if self.direction == "UP":
            danger_left = [head_x - BLOCK, head_y]
            danger_straight = [head_x, head_y - BLOCK]
            danger_right = [head_x + BLOCK, head_y]

        if self.direction == "DOWN":
            danger_left = [head_x + BLOCK, head_y]
            danger_straight = [head_x, head_y + BLOCK]
            danger_right = [head_x - BLOCK, head_y]

        if self.direction == "RIGHT":
            danger_left = [head_x, head_y - BLOCK]
            danger_straight = [head_x + BLOCK, head_y]
            danger_right = [head_x, head_y + BLOCK]

        if self.direction == "LEFT":
            danger_left = [head_x, head_y + BLOCK]
            danger_straight = [head_x - BLOCK, head_y]
            danger_right = [head_x, head_y - BLOCK]

        danger = []

        danger.append(is_crash(danger_left[0], danger_left[1], self.snake))
        danger.append(is_crash(danger_straight[0], danger_straight[1], self.snake))
        danger.append(is_crash(danger_right[0], danger_right[1], self.snake))

        general_state = []

        general_state.append(head_x > self.food[0])
        general_state.append(head_x < self.food[0])
        general_state.append(head_y > self.food[1])
        general_state.append(head_y < self.food[1])

        if self.direction == "RIGHT":
            general_state = [
                general_state[2],
                general_state[3],
                general_state[1],
                general_state[0],
            ]
        if self.direction == "DOWN":
            general_state = [
                general_state[1],
                general_state[0],
                general_state[3],
                general_state[2],
            ]
        if self.direction == "LEFT":
            general_state = [
                general_state[3],
                general_state[2],
                general_state[0],
                general_state[1],
            ]

        state = danger + general_state

        return state

    def decode_turn(self, action):
        index = COMPASS.index(self.direction)

        if action[1] == 1:
            pass
        elif action[0] == 1:
            self.direction = COMPASS[index - 1]
        elif action[2] == 1:
            self.direction = COMPASS[(index + 1) % len(COMPASS)]


class Renderer:
    def __init__(self, enabled=True):
        pygame.init()
        self.enabled = enabled
        random_colour = random.Random()
        self.snake_colour = (
            random_colour.randint(0, 255),
            random_colour.randint(0, 255),
            random_colour.randint(0, 255),
        )
        self.switch_display()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def draw_frame(self, game):
        self.display.fill("white")

        for x_position, y_position in game.snake:
            pygame.draw.rect(
                self.display,
                self.snake_colour,
                pygame.Rect(x_position, y_position, BLOCK, BLOCK),
                width=4,
            )

        pygame.draw.rect(
            self.display, "black", pygame.Rect(game.food[0], game.food[1], BLOCK, BLOCK)
        )

        score_text = self.font.render(f"Score: {game.score}", True, "black")
        self.display.blit(score_text, (25, 25))

        pygame.display.flip()
        self.clock.tick(FPS)

    def switch_display(self):
        if self.enabled:
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.display = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.HIDDEN
            )

    def switch_render(self):
        self.enabled = not self.enabled
        self.switch_display()


class Agent:
    def __init__(self, config, randomizer):
        self.model = nn.Sequential(
            nn.Linear(7, config["inner_layer"]),
            nn.ReLU(),
            nn.Linear(config["inner_layer"], 3),
        )
        self.optimizer = optim.Adam(self.model.parameters(), config["learning_rate"])
        self.criterion = nn.MSELoss()

        self.randomizer = randomizer

    def choose_action(self, state, count, config):
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            prediction = self.model(state0)
        move = torch.argmax(prediction).item()

        if config["exploration_games"] - count > self.randomizer.randint(
            0, config["exploration_games"]
        ):
            move = self.randomizer.randint(0, 2)

        final_move[move] = 1
        return final_move

    def train(self, state, action, next_state, reward, is_game_over, config):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        is_game_over = torch.tensor(is_game_over, dtype=torch.float).unsqueeze(0)

        pred = self.model(state)
        target = pred.clone()

        q_new = reward[0]
        if not is_game_over[0]:
            q_new = (
                reward[0]
                + config["trainer_gamma"] * torch.max(self.model(next_state)).detach()
            )

        target[0][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Plot:
    def __init__(self, enabled=True):
        self.score_series = []
        self.smooth_score_series = []
        self.mean_score_series = []
        self.enabled = enabled
        self.init()

    def init(self):
        if self.enabled:
            plt.ion()

    def update(self):
        plt.clf()
        plt.plot(self.score_series, color="b", lw=1)
        plt.plot(self.mean_score_series, color="r", lw=2)
        plt.plot(self.smooth_score_series, color="orange", lw=2)
        plt.title("Learning...")
        plt.xlabel("Game number")
        plt.ylabel("Score")
        plt.grid(False)
        plt.ylim(0, PLOT_MAX_SCORE)

    def draw(self):
        plt.draw()
        plt.pause(0.001)

    def refresh_score(self, score):
        self.score_series.append(score)
        self.mean_score_series.append(sum(self.score_series) / len(self.score_series))
        self.smooth_score_series.append(
            sum(self.score_series[-50:]) / len(self.score_series[-50:])
        )

    def save(self, run_file):
        self.update()
        plt.savefig(run_file)

    def get_stats(self):
        self.max_score = max(self.score_series)
        self.final_mean_score = sum(self.score_series[-100:]) / len(
            self.score_series[-100:]
        )


class RunPaths:
    def __init__(self, run_type):
        self.single_runs_dir = "./runs/single_runs"
        self.manual_sweeps_dir = "./runs/manual_sweeps"
        self.GA_dir = "./runs/GA"

        if not os.path.exists(self.single_runs_dir):
            os.makedirs(self.single_runs_dir)
        if not os.path.exists(self.manual_sweeps_dir):
            os.makedirs(self.manual_sweeps_dir)
        if not os.path.exists(self.GA_dir):
            os.makedirs(self.GA_dir)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_type = run_type

        if self.run_type == "single_run":
            self.experiment_dir = self.single_runs_dir
        elif self.run_type == "manual_sweep":
            self.experiment_dir = f"{self.manual_sweeps_dir}/{self.timestamp}"
        elif self.run_type == "GA":
            self.experiment_dir = f"{self.GA_dir}/{self.timestamp}"

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def get_run_file_name(
        self, seed=None, config_name=None, run_number=None, generation=None
    ):
        if self.run_type == "single_run":
            self.run_file = f"{self.experiment_dir}/{self.timestamp}"

        elif self.run_type == "manual_sweep":
            self.run_name = f"run{run_number}_seed{seed}_{config_name}"
            self.run_file = f"{self.experiment_dir}/{self.run_name}"

        elif self.run_type == "GA":
            self.run_name = f"gen{generation}_run{run_number}_seed{seed}"
            self.run_file = f"{self.experiment_dir}/{self.run_name}"

        return self.run_file


def is_crash(x, y, snake):
    if x >= SCREEN_WIDTH or y >= SCREEN_HEIGHT or x < 0 or y < 0:
        return True

    if [x, y] in snake[1:-1]:
        return True
    else:
        return False


def spawn_food(snake, randomizer):
    while True:
        food = [
            randomizer.randint(0, SCREEN_WIDTH // BLOCK - 1) * BLOCK,
            randomizer.randint(0, SCREEN_HEIGHT // BLOCK - 1) * BLOCK,
        ]
        if food not in snake:
            break

    return food


def set_seed(config):
    if "seed" in config:
        seed = config["seed"]
    else:
        seed = random.SystemRandom().randint(0, 100000)
        config["seed"] = seed

    randomizer = random.Random(seed)
    torch.manual_seed(seed)

    return randomizer


def get_limit_games(config):
    if "games_after_exploration" in config:
        config["limit_games"] = (
            config["exploration_games"] + config["games_after_exploration"]
        )
    else:
        config["limit_games"] = None


def run(config, game, renderer, agent, plot):
    while True:
        if renderer is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        renderer.switch_render()

                    if event.key == pygame.K_BACKSPACE:
                        plot.refresh_score(game.score)
                        game.reset()

        state = game.get_state()
        action = agent.choose_action(state, game.count, config)

        is_game_over, reward = game.step(action, config)

        if renderer is not None and renderer.enabled:
            renderer.draw_frame(game)

        next_state = game.get_state()

        agent.train(state, action, next_state, reward, is_game_over, config)

        if is_game_over:
            plot.refresh_score(game.score)

            if (
                config["limit_games"] is not None
                and game.count >= config["limit_games"]
            ):
                plot.get_stats()
                return

            if game.count % PLOT_RATE == 0 and plot.enabled:
                plot.update()
                plot.draw()

            game.reset()


def main():
    with open("config.toml", "rb") as file:
        config = tomllib.load(file)
        get_limit_games(config)

    randomizer = set_seed(config)

    game = SnakeGame(randomizer)
    renderer = Renderer()
    agent = Agent(config, randomizer)
    run_paths = RunPaths("single_run")
    plot = Plot(enabled=True)

    run(config, game, renderer, agent, plot)
    run_file = run_paths.get_run_file_name()
    plot.save(run_file)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
