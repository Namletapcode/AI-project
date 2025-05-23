import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
import math
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Configuration
RESULTS_DIR = "/content/drive/MyDrive/game_ai"
NUM_RUNS = 10
NUM_THREADS = 4


class BenchmarkRunner:
    def __init__(self, num_runs=NUM_RUNS, num_threads=NUM_THREADS):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []
        
    def run_benchmark(self, algorithms):
        """Run benchmark for all algorithms"""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for name, algo in algorithms.items():
                for i in range(self.num_runs):
                    futures.append(executor.submit(
                        self._run_single_test, name, algo, i+1
                    ))

            for future in futures:
                result = future.result()
                if result:
                    self.results.append(result)
        
        return pd.DataFrame(self.results)

    def _run_single_test(self, name, algorithm, run_num):
        """Run a single test for an algorithm"""
        try:
            game = Game()
            bot_manager = BotManager(game)
            bot = bot_manager.create_bot(algorithm)
            
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            start_time = time.time()
            score = self._run_game_loop(game, bot_manager)
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

    def _run_game_loop(self, game, bot_manager):
        """Run the game loop for a single test"""
        while not game.game_over:
            state = self._get_game_state(game, bot_manager)
            action = bot_manager.get_action(state)
            game.update(action)
        return game.score

    def _get_game_state(self, game, bot_manager):
        """Get properly formatted game state for the bot"""
        if bot_manager.is_heuristic:
            # For heuristic bots - return list of bullet dicts with x,y
            bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
            bullet_list = []
            for bullet in bullets:
                if hasattr(bullet, 'x') and hasattr(bullet, 'y'):
                    bullet_list.append({'x': bullet.x, 'y': bullet.y})
                elif hasattr(bullet, 'centerx') and hasattr(bullet, 'centery'):
                    bullet_list.append({'x': bullet.centerx, 'y': bullet.centery})
            return bullet_list
        
        elif bot_manager.is_vision:
            # For vision-based bots
            return game.get_screen_image()
        
        else:
            # For parametric bots (deep learning param input)
            player = game.player
            bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
            bullet_data = []
            for bullet in bullets:
                x = bullet.x if hasattr(bullet, 'x') else bullet.centerx
                y = bullet.y if hasattr(bullet, 'y') else bullet.centery
                vx = vy = 0
                if hasattr(bullet, 'angle') and hasattr(bullet, 'speed'):
                    vx = math.cos(bullet.angle) * bullet.speed
                    vy = math.sin(bullet.angle) * bullet.speed
                bullet_data.extend([
                    (x - player.x) / SCAN_RADIUS,
                    (y - player.y) / SCAN_RADIUS,
                    vx / 10.0,
                    vy / 10.0
                ])
            max_bullets = 20
            bullet_data = bullet_data[:max_bullets * 4] + [0] * (max_bullets * 4 - len(bullet_data))
            
            return np.array([
                (player.x - BOX_LEFT) / BOX_SIZE,
                (player.y - BOX_TOP) / BOX_SIZE,
                *bullet_data
            ], dtype=np.float32)


def setup_environment():
    """Set up headless pygame environment"""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))

def save_visualizations(df):
    """Save visualizations to drive"""
    timestamp = int(time.time())
    
    plt.figure(figsize=(12, 6))
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        # Trung bình điểm theo run (hoặc run là thứ tự test)
        mean_scores = algo_df.groupby('run')['score'].mean()
        plt.plot(mean_scores.index, mean_scores.values, 'o-', label=algo)
    
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
    from configs.bot_config import DodgeAlgorithm, SCAN_RADIUS
    from configs.game_config import BOX_LEFT, BOX_TOP, BOX_SIZE

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
    print(f"Benchmark completed in {time.time()-start_time:.2f} seconds")
