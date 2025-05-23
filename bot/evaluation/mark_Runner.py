import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
import math
import sys
from types import SimpleNamespace


# Setup paths
project_root = '/content/AI-project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration first
from configs.game_config import BOX_LEFT, BOX_TOP, BOX_SIZE
from configs.bot_config import DodgeAlgorithm, SCAN_RADIUS

# Configuration
RESULTS_DIR = "/content/drive/MyDrive/game_ai"
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_RUNS = 10
NUM_THREADS = 4

class BenchmarkRunner:
    def __init__(self, num_runs=NUM_RUNS, num_threads=NUM_THREADS):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []

    def run_benchmark(self, algorithms):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for name, algo in algorithms.items():
                for i in range(self.num_runs):
                    futures.append(executor.submit(
                        self._run_single_test, name, algo, i + 1
                    ))

            for future in futures:
                result = future.result()
                if result:
                    self.results.append(result)

        return pd.DataFrame(self.results)

    def _run_single_test(self, name, algorithm, run_num):
        try:
            game = Game()
            bot_manager = BotManager(game)
            bot = bot_manager.create_bot(algorithm)

            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            if hasattr(bot, 'get_action'):
                if 'vision' in algorithm.name.lower():
                    bot.is_vision = True
                    bot.is_parametric = False
                    bot.is_heuristic = False
                elif 'param' in algorithm.name.lower():
                    bot.is_parametric = True
                    bot.is_vision = False
                    bot.is_heuristic = False
                else:
                    bot.is_heuristic = True
                    bot.is_vision = False
                    bot.is_parametric = False

            start_time = time.time()
            score = self._run_game_loop(game, bot_manager, bot)
            duration = time.time() - start_time

            return {
                "algorithm": name,
                "run": run_num,
                "score": score,
                "duration": duration
            }

        except Exception as e:
            print(f"Error in {name} run {run_num}: {str(e)}")
            return None

    def _run_game_loop(self, game, bot_manager, bot):
        try:
            while not game.game_over:
                state = self._get_game_state(game, bot_manager, bot)

                if hasattr(bot, 'is_heuristic') and bot.is_heuristic:
                    action = bot.get_action(state)

                elif hasattr(bot, 'is_parametric') and bot.is_parametric:
                    if hasattr(bot, 'eval'):
                        action = bot.eval(state)
                    elif hasattr(bot, 'get_action'):
                        action = bot.get_action(state)
                    else:
                        raise ValueError("Parametric bot missing action method")

                elif hasattr(bot, 'is_vision') and bot.is_vision:
                    if hasattr(bot, 'eval'):
                        action = bot.eval(state)
                    elif hasattr(bot, 'get_action'):
                        action = bot.get_action(state)
                    else:
                        raise ValueError("Vision bot missing action method")

                else:
                    raise ValueError("Unknown bot type or unsupported bot interface")

                game.update(action)

            return game.score
        except Exception as e:
            print(f"Game loop error: {str(e)}")
            return 0

    def _get_game_state(self, game, bot_manager, bot):
        try:
            if hasattr(bot, 'is_heuristic') and bot.is_heuristic:
                return self._get_heuristic_state(game)
            elif hasattr(bot, 'is_vision') and bot.is_vision:
                return self._get_vision_state(game)
            elif hasattr(bot, 'is_parametric') and bot.is_parametric:
                return self._get_parametric_state(game)
            else:
                raise ValueError("Bot type not specified correctly (heuristic/vision/parametric)")
        except Exception as e:
            print(f"State generation error: {str(e)}")
            return np.zeros(82, dtype=np.float32)

    def _get_heuristic_state(self, game):
        bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
        bullet_objects = []
        for bullet in bullets:
            try:
                x = getattr(bullet, 'x', getattr(bullet, 'centerx', 0.0))
                y = getattr(bullet, 'y', getattr(bullet, 'centery', 0.0))
                bullet_objects.append(SimpleNamespace(x=float(x), y=float(y)))
            except Exception as e:
                continue
        return bullet_objects

    def _get_vision_state(self, game):
        try:
            if hasattr(game, 'get_screen_image'):
                return game.get_screen_image()
            return np.zeros((600, 800, 3), dtype=np.float32)
        except:
            return np.zeros((600, 800, 3), dtype=np.float32)

    def _get_parametric_state(self, game):
        player = game.player
        bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)

        bullet_data = []
        for bullet in bullets:
            try:
                x = getattr(bullet, 'x', getattr(bullet, 'centerx', 0.0))
                y = getattr(bullet, 'y', getattr(bullet, 'centery', 0.0))
                angle = getattr(bullet, 'angle', 0.0)
                speed = getattr(bullet, 'speed', 0.0)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                bullet_data.extend([
                    (x - player.x) / SCAN_RADIUS,
                    (y - player.y) / SCAN_RADIUS,
                    vx / 10.0,
                    vy / 10.0
                ])
            except:
                continue

        max_bullets = 20
        bullet_data = bullet_data[:max_bullets * 4] + [0] * (max_bullets * 4 - len(bullet_data))

        return np.array([
            (player.x - BOX_LEFT) / BOX_SIZE,
            (player.y - BOX_TOP) / BOX_SIZE,
            *bullet_data
        ], dtype=np.float32)

def setup_environment():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        pygame.init()
        pygame.display.set_mode((1, 1))
    except:
        print("PyGame initialization warning - continuing in headless mode")

def save_visualizations(df):
    timestamp = int(time.time())

    plt.figure(figsize=(12, 6))
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        plt.plot(algo_df['run'], algo_df['score'], 'o-', label=algo)

    plt.title("Algorithm Performance Comparison")
    plt.xlabel("Run Number")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{RESULTS_DIR}/performance_comparison_{timestamp}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    csv_path = f"{RESULTS_DIR}/benchmark_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Results saved to:\n- {plot_path}\n- {csv_path}")

if __name__ == "__main__":
    from game.game_core import Game
    from bot.bot_manager import BotManager

    setup_environment()

    algorithms = {
        "Furthest Safe": DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        "Least Danger": DodgeAlgorithm.LEAST_DANGER_PATH,
        "Least Danger Advanced": DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        "Opposite Threat": DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
        "Random Safe Zone": DodgeAlgorithm.RANDOM_SAFE_ZONE,
        "DL Param (Numpy)": DodgeAlgorithm.DL_PARAM_INPUT_NUMPY,
        "DL Param (Torch)": DodgeAlgorithm.DL_PARAM_INPUT_TORCH,
        "DL Vision (Numpy)": DodgeAlgorithm.DL_VISION_INPUT_NUMPY
    }

    print("Starting benchmark...")
    start_time = time.time()

    benchmark = BenchmarkRunner()
    results_df = benchmark.run_benchmark(algorithms)

    save_visualizations(results_df)

    pygame.quit()
    print(f"Benchmark completed in {time.time() - start_time:.2f} seconds")
