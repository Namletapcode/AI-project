import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
from types import SimpleNamespace

project_root = '/content/AI-project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm

def one_hot_to_vector(action):
    mapping = {
        0: pygame.Vector2(-1, -1),
        1: pygame.Vector2(0, -1),
        2: pygame.Vector2(1, -1),
        3: pygame.Vector2(-1, 0),
        4: pygame.Vector2(0, 0),
        5: pygame.Vector2(1, 0),
        6: pygame.Vector2(-1, 1),
        7: pygame.Vector2(0, 1),
        8: pygame.Vector2(1, 1),
    }
    if isinstance(action, (list, np.ndarray)):
        index = int(np.argmax(action))
        return mapping.get(index, pygame.Vector2(0, 0))
    return pygame.Vector2(0, 0)

class HeadlessBenchmark:
    def __init__(self, num_runs=1, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []

    def _run_single_test(self, name, algorithm, run_idx):
        try:
            game = Game()
            bot_manager = BotManager(game)

            bot_creators = {
                DodgeAlgorithm.FURTHEST_SAFE_DIRECTION: lambda: bot_manager.create_bot(DodgeAlgorithm.FURTHEST_SAFE_DIRECTION),
                DodgeAlgorithm.LEAST_DANGER_PATH: lambda: bot_manager.create_bot(DodgeAlgorithm.LEAST_DANGER_PATH),
                DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED: lambda: bot_manager.create_bot(DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED),
                DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION: lambda: bot_manager.create_bot(DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION),
                DodgeAlgorithm.RANDOM_SAFE_ZONE: lambda: bot_manager.create_bot(DodgeAlgorithm.RANDOM_SAFE_ZONE),
                DodgeAlgorithm.DL_PARAM_INPUT_NUMPY: lambda: bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_NUMPY),
                DodgeAlgorithm.DL_PARAM_INPUT_TORCH: lambda: bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_TORCH),
                DodgeAlgorithm.DL_VISION_INPUT_NUMPY: lambda: bot_manager.create_bot(DodgeAlgorithm.DL_VISION_INPUT_NUMPY)
            }

            bot = bot_creators.get(algorithm, lambda: None)()
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            is_heuristic = getattr(bot, "is_heuristic", False)

            while True:
                state = game.get_state()

                if is_heuristic:
                    if isinstance(state, dict):
                        state = SimpleNamespace(**state)

                    # Convert bullets to pygame.Vector2
                    if hasattr(state, 'bullets'):
                        processed_bullets = []
                        for bullet in state.bullets:
                            try:
                                if isinstance(bullet, pygame.Vector2):
                                    processed_bullets.append(bullet)
                                elif isinstance(bullet, (list, tuple, np.ndarray)) and len(bullet) == 2:
                                    processed_bullets.append(pygame.Vector2(float(bullet[0]), float(bullet[1])))
                                elif hasattr(bullet, 'x') and hasattr(bullet, 'y'):
                                    processed_bullets.append(pygame.Vector2(float(bullet.x), float(bullet.y)))
                            except Exception as e:
                                print(f"[WARNING] Failed to convert bullet: {bullet} -> {e}")
                        state.bullets = processed_bullets

                    action = bot.get_action(state)
                else:
                    action = bot.get_action(state)
                    if isinstance(action, (list, np.ndarray)) and len(action) == 9:
                        action = one_hot_to_vector(action)

                game.update(action)

                if game.game_over:
                    break

            return {
                "algorithm": name,
                "run": run_idx + 1,
                "score": game.score,
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] Run failed for {name} (algo={algorithm}): {e}")
            traceback.print_exc()
            return None

    def run(self, algorithms):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for name, algo in algorithms.items():
                for i in range(self.num_runs):
                    futures.append(executor.submit(
                        self._run_single_test, name, algo, i
                    ))

            for future in futures:
                if (result := future.result()):
                    self.results.append(result)

        return pd.DataFrame(self.results)

def setup_environment():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))

def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    os.makedirs(base_path, exist_ok=True)
    if df.empty:
        return None, None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Plotting
    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)

    algorithms = df['algorithm'].unique()
    plot_paths = []
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo].copy()
        plt.figure(figsize=(10, 6))
        plt.plot(algo_df['run'], algo_df['score'], marker='o', color='blue')
        plt.title(f"Performance of {algo} (Raw Scores)", fontsize=16)
        plt.xlabel("Run Number", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.grid(True)
        plot_path = os.path.join(plots_dir, f"{algo.replace(' ', '_')}_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)

    # Combined plot
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(right=0.75)
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo].copy()
        algo_df['cumulative_avg'] = algo_df['score'].expanding().mean()
        plt.plot(algo_df['run'], algo_df['cumulative_avg'], label=algo)
    plt.title("Algorithm Comparison (Cumulative Averages)", fontsize=16)
    plt.xlabel("Number of Runs", fontsize=14)
    plt.ylabel("Cumulative Average Score", fontsize=14)
    plt.grid(True)
    plt.legend(title="Algorithms", fontsize=12,
               bbox_to_anchor=(1.05, 1), loc='upper left')
    combined_plot_path = f"{base_path}/combined_plot.png"
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()

    return csv_path, plot_paths, combined_plot_path

if __name__ == "__main__":
    setup_environment()

    algorithms = {
        "Furthest Safe": DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        "Least Danger": DodgeAlgorithm.LEAST_DANGER_PATH,
        "Least Danger Advanced": DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        "Opposite Threat Direction": DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
        "Random Safe Zone": DodgeAlgorithm.RANDOM_SAFE_ZONE,
        "DL Numpy": DodgeAlgorithm.DL_PARAM_INPUT_NUMPY,
        "DL Param Torch": DodgeAlgorithm.DL_PARAM_INPUT_TORCH,
        "DL Vision Input Numpy": DodgeAlgorithm.DL_VISION_INPUT_NUMPY,
    }

    benchmark = HeadlessBenchmark(num_runs=20, num_threads=4)
    results_df = benchmark.run(algorithms)

    csv_file, individual_plots, combined_plot = save_results(results_df)
    pygame.quit()
