
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pygame
import numpy as np
from types import SimpleNamespace
from google.colab import drive



# Set up project root
project_root = '/content/AI-project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import game components
from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm, SCAN_RADIUS, BOX_LEFT, BOX_TOP, BOX_SIZE

class ColabBenchmark:
    def __init__(self, num_runs=5, num_threads=4):
        self.num_runs = num_runs
        self.num_threads = num_threads
        self.results = []
        self.live_stats = []
        
    def _run_single_test(self, name, algorithm, run_idx):
        try:
            game = Game(render=False)
            bot_manager = BotManager(game)

            # Create bot based on algorithm
            bot = bot_manager.create_bot(algorithm)
            if not bot:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            start_time = time.time()
            score = self._run_game_loop(game, bot_manager, bot)
            duration = time.time() - start_time

            result = {
                "algorithm": name,
                "run": run_idx + 1,
                "score": score,
                "duration": duration
            }
            
            # Calculate running average
            current_results = [r for r in self.results if r["algorithm"] == name]
            current_results.append(result)
            avg_score = np.mean([r["score"] for r in current_results])
            
            # Update live stats
            self.live_stats.append({
                "Algorithm": name,
                "Run": run_idx + 1,
                "Score": score,
                "Avg": avg_score
            })
            
            # Print progress
            self._print_progress()
            
            return result
            
        except Exception as e:
            print(f"Error in {name} run {run_idx + 1}: {str(e)}")
            return None

    def _run_game_loop(self, game, bot_manager, bot):
        while not game.game_over:
            state = self._get_game_state(game, bot_manager)
            action = bot.get_action(state)
            game.update(action)
            
            if game.game_over:
                break
        return game.score

    def _get_game_state(self, game, bot_manager):
        """Get properly formatted game state for each bot type"""
        current_bot = bot_manager.current_bot
        
        # For heuristic bots - return numpy array of bullet positions
        if hasattr(current_bot, 'is_heuristic') and current_bot.is_heuristic:
            bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
            bullet_array = []
            
            for bullet in bullets:
                # Handle both pygame Rect and custom bullet objects
                if hasattr(bullet, 'x'):
                    x, y = bullet.x, bullet.y
                else:
                    x, y = bullet.centerx, bullet.centery
                
                bullet_array.append([x, y])
            
            return np.array(bullet_array, dtype=np.float32)
        
        # For vision-based bots
        elif hasattr(current_bot, 'is_vision') and current_bot.is_vision:
            return game.get_screen_image()
        
        # For parametric bots
        else:
            return self._get_parametric_state(game)

    def _get_parametric_state(self, game):
        """Create normalized parametric state vector"""
        player = game.player
        bullets = game.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
        
        # Normalize bullet data
        bullet_data = []
        for bullet in bullets:
            if hasattr(bullet, 'x'):
                x, y = bullet.x, bullet.y
            else:
                x, y = bullet.centerx, bullet.centery
                
            if hasattr(bullet, 'angle') and hasattr(bullet, 'speed'):
                vx = math.cos(bullet.angle) * bullet.speed
                vy = math.sin(bullet.angle) * bullet.speed
            else:
                vx, vy = 0, 0
                
            bullet_data.extend([
                (x - player.x) / SCAN_RADIUS,  # Normalized to [-1, 1]
                (y - player.y) / SCAN_RADIUS,
                vx / 10.0,  # Assuming max speed of 10
                vy / 10.0
            ])
        
        # Limit bullets and pad with zeros
        max_bullets = 20
        bullet_data = bullet_data[:max_bullets * 4]
        bullet_data.extend([0] * (max_bullets * 4 - len(bullet_data)))
        
        # Add normalized player position
        state = [
            (player.x - BOX_LEFT) / BOX_SIZE,
            (player.y - BOX_TOP) / BOX_SIZE,
            *bullet_data
        ]
        
        return np.array(state, dtype=np.float32)

    def _print_progress(self):
        """Print live updating table of results"""
        from IPython.display import clear_output
        import pandas as pd
        
        clear_output(wait=True)
        
        if not self.live_stats:
            print("No results yet...")
            return
            
        # Create DataFrame from live stats
        df = pd.DataFrame(self.live_stats)
        
        # Pivot table for better display
        pivot_df = df.pivot(index='Run', columns='Algorithm', values=['Score', 'Avg'])
        
        # Display the table
        display(pivot_df.style.set_caption("Live Benchmark Results"))
        
        # Also show the latest raw scores
        print("\nLatest Scores:")
        display(df.tail(10))

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
    """Set up headless pygame environment for Colab"""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))

def save_and_visualize_results(df, base_path="/content/drive/MyDrive/game_ai"):
    """Save results and create visualizations"""
    from IPython.display import display, HTML
    
    # Create directory if not exists
    os.makedirs(base_path, exist_ok=True)
    
    if df.empty:
        print("No results to save")
        return

    # Save raw results
    csv_path = f"{base_path}/benchmark_results_{int(time.time())}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Create visualizations
    plt.style.use('seaborn')
    
    # 1. Individual algorithm performance
    print("\nIndividual Algorithm Performance:")
    algorithms = df['algorithm'].unique()
    
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(12, 3*len(algorithms)))
    if len(algorithms) == 1:
        axes = [axes]
    
    for idx, algo in enumerate(algorithms):
        algo_df = df[df['algorithm'] == algo].copy()
        algo_df['running_avg'] = algo_df['score'].expanding().mean()
        
        ax = axes[idx]
        ax.plot(algo_df['run'], algo_df['score'], 'o-', label='Score', alpha=0.5)
        ax.plot(algo_df['run'], algo_df['running_avg'], 'r-', label='Running Avg')
        
        ax.set_title(f"{algo} Performance", fontsize=12)
        ax.set_xlabel("Run Number")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    individual_plot_path = f"{base_path}/individual_performance.png"
    plt.savefig(individual_plot_path)
    plt.show()
    
    # 2. Combined comparison
    print("\nAlgorithm Comparison:")
    plt.figure(figsize=(14, 8))
    
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo].copy()
        algo_df['running_avg'] = algo_df['score'].expanding().mean()
        plt.plot(algo_df['run'], algo_df['running_avg'], label=algo, linewidth=2)
    
    plt.title("Algorithm Comparison (Running Averages)", fontsize=16)
    plt.xlabel("Number of Runs", fontsize=14)
    plt.ylabel("Running Average Score", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    combined_plot_path = f"{base_path}/algorithm_comparison.png"
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.show()
    
    # 3. Final summary table
    print("\nFinal Summary:")
    summary_df = df.groupby('algorithm')['score'].agg(['mean', 'std', 'min', 'max'])
    display(summary_df.style.background_gradient(cmap='Blues'))
    
    # Save summary
    summary_path = f"{base_path}/summary_stats.csv"
    summary_df.to_csv(summary_path)
    
    return {
        'csv': csv_path,
        'individual_plot': individual_plot_path,
        'combined_plot': combined_plot_path,
        'summary': summary_path
    }

if __name__ == "__main__":
    # Set up environment
    setup_environment()

    # Define algorithms to test
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

    # Run benchmark
    print("Starting benchmark...")
    benchmark = ColabBenchmark(num_runs=10, num_threads=4)
    results_df = benchmark.run(algorithms)

    # Save and visualize results
    results = save_and_visualize_results(results_df)

    # Clean up
    pygame.quit()
    print("Benchmark completed!")
