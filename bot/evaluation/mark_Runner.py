import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
from types import SimpleNamespace

project_root = '/content/e-project'
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
        
    def _create_bot_lambda(self, algorithm):
        """Tạo bot sử dụng lambda function"""
        return lambda game: BotManager(game).create_bot(algorithm)

    def _process_bullets(self, bullets):
        """Xử lý bullets với lambda function"""
        return [
            (lambda b: pygame.Vector2(
                float(b[0]) if isinstance(b, (list, tuple, np.ndarray)) else float(b.x),
                float(b[1]) if isinstance(b, (list, tuple, np.ndarray)) else float(b.y)
            ))(bullet)
            for bullet in bullets
            if (hasattr(bullet, 'x') and hasattr(bullet, 'y')) or 
               (isinstance(bullet, (list, tuple, np.ndarray)) and len(bullet) >= 2)
        ]

    def _run_single_test(self, name, algorithm, run_idx):
        try:
            game = Game()
            create_bot = self._create_bot_lambda(algorithm)
            bot = create_bot(game)
            
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            start_time = time.time()
            while True:
                state = game.get_state()
                
                if getattr(bot, "is_heuristic", False):
                    bullets = (
                        state.bullets if hasattr(state, 'bullets') else
                        state['bullets'] if isinstance(state, dict) else []
                    )
                    processed_bullets = self._process_bullets(bullets)
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
        except Exception as e:
            print(f"Error in {name} run {run_idx + 1}: {str(e)}")
            return None

    def run(self, algorithms):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self._run_single_test, name, algo, i)
                for name, algo in algorithms.items()
                for i in range(self.num_runs)
            ]
            
            self.results = [
                future.result() for future in futures 
                if future.result() is not None
            ]

        return pd.DataFrame(self.results)

def setup_environment():
    """Cấu hình môi trường pygame headless"""
    os.environ.update({
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy"
    })
    pygame.init()
    pygame.display.set_mode((1, 1))

def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    """Lưu kết quả và tạo biểu đồ"""
    os.makedirs(base_path, exist_ok=True)
    if df.empty:
        return None, [], None

    # Lưu CSV
    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Tạo thư mục cho các biểu đồ riêng lẻ
    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Tạo biểu đồ cho từng thuật toán
    plot_paths = []
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        
        plt.figure(figsize=(10, 6))
        plt.plot(algo_df['run'], algo_df['score'], 'o-', label=algo)
        plt.title(f"Performance: {algo}")
        plt.xlabel("Run Number")
        plt.ylabel("Score")
        plt.grid(True)
        
        plot_path = os.path.join(plots_dir, f"{algo.replace(' ', '_')}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)

    # Tạo biểu đồ tổng hợp
    plt.figure(figsize=(14, 8))
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        plt.plot(algo_df['run'], algo_df['score'], 'o-', label=algo)
    
    plt.title("Algorithm Comparison")
    plt.xlabel("Run Number")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    combined_path = f"{base_path}/combined_results.png"
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close()

    return csv_path, plot_paths, combined_path

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

    benchmark = HeadlessBenchmark(num_runs=50, num_threads=4)
    results_df = benchmark.run(algorithms)

    csv_file, individual_plots, combined_plot = save_results(results_df)
    pygame.quit()
