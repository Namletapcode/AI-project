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

def process_bullets(bullets):
    """Chuyển đổi danh sách đạn sang SimpleNamespace có thuộc tính x,y"""
    processed = []
    for bullet in bullets:
        if hasattr(bullet, 'x') and hasattr(bullet, 'y'):
            # Nếu bullet đã có thuộc tính x,y thì giữ nguyên
            processed.append(bullet)
        elif isinstance(bullet, (list, tuple, np.ndarray)) and len(bullet) >= 2:
            # Nếu là mảng thì chuyển thành SimpleNamespace
            bullet_ns = SimpleNamespace()
            bullet_ns.x = float(bullet[0])
            bullet_ns.y = float(bullet[1])
            processed.append(bullet_ns)
        elif isinstance(bullet, dict) and 'x' in bullet and 'y' in bullet:
            # Nếu là dict thì chuyển thành SimpleNamespace
            bullet_ns = SimpleNamespace()
            bullet_ns.x = float(bullet['x'])
            bullet_ns.y = float(bullet['y'])
            processed.append(bullet_ns)
    return processed

class HeadlessBenchmark:
    def __init__(self, num_runs=1, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []
        
    def _run_single_test(self, name, algorithm, run_idx):
        try:
            game = Game()
            bot_manager = BotManager(game)

            bot = bot_manager.create_bot(algorithm)
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
            is_heuristic = getattr(bot, "is_heuristic", False)
            start_time = time.time()
            
            while True:
                if is_heuristic:
                    # Lấy đạn và xử lý thành SimpleNamespace có x,y
                    bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
                    processed_bullets = process_bullets(bullets)
                    
                    # Tạo state phù hợp cho heuristic bot
                    state = SimpleNamespace(
                        bullets=processed_bullets,
                        player=SimpleNamespace(
                            x=float(game.player.x),
                            y=float(game.player.y),
                            velocity_x=0,
                            velocity_y=0,
                            directions=game.player.directions,
                            direction_to_position=game.player.direction_to_position
                        )
                    )
                    action = bot.get_action(state)
                else:
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
        print("No results to save!")
        return None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    plt.figure(figsize=(14, 8))
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        plt.plot(algo_df['run'], algo_df['score'], 'o-', label=algo)
    
    plt.title("Algorithm Performance Comparison")
    plt.xlabel("Run Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    
    plot_path = f"{base_path}/performance_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    return csv_path, plot_path

if __name__ == "__main__":
    print("Setting up environment...")
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

    print("Starting benchmark...")
    benchmark = HeadlessBenchmark(num_runs=20, num_threads=4)
    results_df = benchmark.run(algorithms)

    print("Saving results...")
    csv_file, plot_file = save_results(results_df)
    
    print("Benchmark completed!")
    pygame.quit()
