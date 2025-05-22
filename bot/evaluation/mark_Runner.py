import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
from types import SimpleNamespace

project_root = '/content/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm

class HeadlessBenchmark:
    def __init__(self, num_runs=5, num_threads=4):
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
                DodgeAlgorithm.DL_PARAM_INPUT_TORCH: lambda: bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_TORCH)
            }

            bot = bot_creators.get(algorithm, lambda: None)()
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            start_time = time.time()
            while True:
                state = game.get_state()
                if getattr(bot, "is_heuristic", False):
                    if isinstance(state, np.ndarray):
                        action = pygame.Vector2(0, 0)
                    else:
                        if hasattr(state, 'bullets'):
                            bullets = state.bullets
                        elif isinstance(state, dict) and 'bullets' in state:
                            bullets = state['bullets']
                        else:
                            bullets = []
                        processed_bullets = []
                        for bullet in bullets:
                            if isinstance(bullet, (list, tuple, np.ndarray)) and len(bullet) == 2:
                                processed_bullets.append(pygame.Vector2(float(bullet[0]), float(bullet[1])))
                            elif hasattr(bullet, 'x') and hasattr(bullet, 'y'):
                                processed_bullets.append(pygame.Vector2(float(bullet.x), float(bullet.y)))

                        action = bot.get_action(processed_bullets)
                else:
                    action = bot.get_action(state)

                game.update(action)
                if game.game_over:
                    break

            return {
                "algorithm": name,
                "run": run_idx + 1,
                "score": game.score,
                "duration": time.time() - start_time
            }
        except Exception:
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
        return None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Create a directory for individual plots
    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get unique algorithms
    algorithms = df['algorithm'].unique()
    
    # Create individual plots for each algorithm (show raw scores)
    plot_paths = []
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo].copy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(algo_df['run'], algo_df['score'], marker='o', color='blue')
        
        plt.title(f"Performance of {algo} (Raw Scores)", fontsize=16)
        plt.xlabel("Run Number", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.grid(True)
        
       
        
        # Save individual plot
        plot_path = os.path.join(plots_dir, f"{algo.replace(' ', '_')}_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)

    # Create a combined plot with cumulative averages
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
    }

    benchmark = HeadlessBenchmark(num_runs=30, num_threads=4)
    results_df = benchmark.run(algorithms)

    csv_file, individual_plots, combined_plot = save_results(results_df)

    if csv_file and individual_plots:
        print("\nBenchmark completed!")
        print(f"→ CSV results: {csv_file}")
        print("→ Individual plots:")
        for plot in individual_plots:
            print(f"   - {plot}")
        print(f"→ Combined plot: {combined_plot}")
        print("\nScore statistics:")
        print(results_df.groupby('algorithm')['score'].describe())

    pygame.quit()
