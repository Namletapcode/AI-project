
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
from configs.bot_config import DodgeAlgorithm, SCAN_RADIUS

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
    def __init__(self, num_runs=5, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []
        
    def _create_heuristic_state(self, game):
        """Tạo state phù hợp cho heuristic bot"""
        bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
        
        # Xử lý bullets thành các object có thuộc tính x,y
        processed_bullets = []
        for bullet in bullets:
            if hasattr(bullet, 'x') and hasattr(bullet, 'y'):
                # Nếu đã là object có x,y thì giữ nguyên
                processed_bullets.append(bullet)
            elif isinstance(bullet, (list, tuple, np.ndarray)) and len(bullet) >= 2:
                # Nếu là mảng thì chuyển thành SimpleNamespace
                bullet_ns = SimpleNamespace()
                bullet_ns.x = float(bullet[0])
                bullet_ns.y = float(bullet[1])
                processed_bullets.append(bullet_ns)
            elif isinstance(bullet, dict) and 'x' in bullet and 'y' in bullet:
                # Nếu là dict thì chuyển thành SimpleNamespace
                bullet_ns = SimpleNamespace()
                bullet_ns.x = float(bullet['x'])
                bullet_ns.y = float(bullet['y'])
                processed_bullets.append(bullet_ns)
        
        # Tạo player state với đầy đủ thuộc tính cần thiết
        player_state = SimpleNamespace(
            x=float(game.player.x),
            y=float(game.player.y),
            directions=game.player.directions,
            direction_to_position=game.player.direction_to_position
        )
        
        return processed_bullets  # HeuristicDodgeBot nhận trực tiếp list bullets

    def _run_single_test(self, name, algorithm, run_idx):
        try:
            game = Game()
            bot_manager = BotManager(game)
            bot = bot_manager.create_bot(algorithm)
            
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            start_time = time.time()
            
            while True:
                if getattr(bot, "is_heuristic", False):
                    # Đối với heuristic bot, truyền danh sách bullets đã xử lý
                    bullets = self._create_heuristic_state(game)
                    action = bot.get_action(bullets)
                else:
                    # Đối với DL bot, truyền state nguyên bản
                    state = game.get_state()
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
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            print(f"Error in {name} run {run_idx + 1}: {str(e)}")
            import traceback
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
        return None, [], None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_paths = []
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].copy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(algo_df['run'], algo_df['score'], marker='o', color='blue')
        plt.title(f"Performance of {algo}")
        plt.xlabel("Run Number")
        plt.ylabel("Score")
        plt.grid(True)
        
        plot_path = os.path.join(plots_dir, f"{algo.replace(' ', '_')}_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)

    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(right=0.75)
    
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].copy()
        algo_df['cumulative_avg'] = algo_df['score'].expanding().mean()
        plt.plot(algo_df['run'], algo_df['cumulative_avg'], label=algo)
    
    plt.title("Algorithm Comparison (Cumulative Averages)")
    plt.xlabel("Number of Runs")
    plt.ylabel("Cumulative Average Score")
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
