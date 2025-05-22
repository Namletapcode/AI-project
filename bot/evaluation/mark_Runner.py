import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
import traceback
from configs.bot_config import SCAN_RADIUS

project_root = '/content/AI-project'
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
        
    def _run_single_test(self, algorithm, run_idx):
        try:
          
            game = Game()  
            bot_manager = BotManager(game)
            
            bot = bot_manager.create_bot(algorithm)
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            start_time = time.time()
            while True:
            
                bullets_in_radius = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
                
                processed_bullets = [pygame.Vector2(bullet.x, bullet.y) for bullet in bullets_in_radius]
                
                action = bot.get_action(processed_bullets)
                
                game.update(action)
                
                # Kiểm tra kết thúc game
                if game.game_over:
                    break

            return {
                "algorithm": algorithm.name,
                "run": run_idx + 1,
                "score": game.score,
                "duration": time.time() - start_time,
                "survival_time": game.time_elapsed
            }
        except Exception as e:
            print(f"[ERROR] Algorithm: {algorithm.name}, Run: {run_idx + 1}")
            traceback.print_exc()
            return None

    def run(self, algorithms):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for algo in algorithms:
                for i in range(self.num_runs):
                    futures.append(executor.submit(
                        self._run_single_test, algo, i
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

def save_results(df, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print(" No results to save!")
        return None, None
    
    # Lưu kết quả CSV
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    # Hiển thị thống kê
    print("\n=== Benchmark Summary ===")
    print(df.groupby('algorithm').agg({
        'score': ['mean', 'std', 'max'],
        'survival_time': ['mean', 'max'],
        'duration': ['mean']
    }))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    df.groupby('algorithm')['score'].mean().sort_values().plot(
        kind='barh', title='Average Scores by Algorithm'
    )
    plt.xlabel('Score')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "performance.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")
    
    return csv_path, plot_path

if __name__ == "__main__":
    print(" Starting benchmark...")
    setup_environment()
    
    # Danh sách thuật toán cần test
    algorithms = [
        DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        DodgeAlgorithm.LEAST_DANGER_PATH,
        DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
        DodgeAlgorithm.RANDOM_SAFE_ZONE,
        DodgeAlgorithm.DL_PARAM_INPUT_NUMPY, 
        DodgeAlgorithm.DL_PARAM_INPUT_TORCH   
    ]
    
    benchmark = HeadlessBenchmark(num_runs=10, num_threads=4)
    results_df = benchmark.run(algorithms)
    
    if not results_df.empty:
        csv_file, plot_file = save_results(results_df)
        print("\n Benchmark completed successfully!")
    else:
        print("\n Benchmark failed - no results collected")
    
    pygame.quit()
