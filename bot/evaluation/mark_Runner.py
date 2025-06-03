import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pygame

USE_COLAB = False
USE_AVERAGE = False
base_path = 'saved_files/benchmark'

if USE_COLAB:
    project_root = '/content/AI-project'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    base_path = "/content/drive/MyDrive/game_ai"
elif __name__ == "__main__":
    # only re-direct below if running this file
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm

def run_benchmark_parallel(algorithms, num_episodes=20):
    try:
        positions, window_size = calculate_window_positions(len(algorithms))
        
        pool = multiprocessing.Pool(processes=len(algorithms))
        args = [(algo, num_episodes, (pos[0], pos[1], window_size)) 
               for algo, pos in zip(algorithms, positions)]
        results = pool.starmap(run_algorithm_episodes, args)
        return [r for algo_results in results for r in algo_results]  # Flatten results
    finally:
        pool.close()
        pool.join()
        pygame.quit()

def run_algorithm_episodes(algorithm, num_episodes, window_config):
    """Run all episodes for a single algorithm"""
    x, y, window_size = window_config
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
    display = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption(f"Bot: {algorithm.name}")
    
    game = Game(render=False)
    results = []
    bot_manager = BotManager(game)
    bot_manager.create_bot(algorithm, load_saved_model=True)

    if not bot_manager.is_heuristic:
        bot_manager.current_bot.set_mode("perform")
        bot_manager.current_bot.load_model()

    for episode in range(num_episodes):
        game.restart_game()
        game_over = False
        while not game_over:
            state = game.get_state(bot_manager.is_heuristic, bot_manager.is_vision, bot_manager.method)
            action_idx = bot_manager.current_bot.get_action_idx(state)
            game.update(action_idx)
            game.draw()
            
            scaled_surface = pygame.transform.scale(game.surface, (window_size, window_size))
            display.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            
            _, game_over = game.get_reward()

        score = game.score
        print(f"{algorithm.name} Episode {episode}, Score: {score}")

        results.append({
            "algorithm": algorithm.name,
            "run": episode + 1,
            "score": score,
        })
    return results

def save_individual_results(df: pd.DataFrame, folder_path: str, use_average: bool = False) -> list:
    """
    Save individual algorithm performance plots
    
    Args:
        df: DataFrame with results
        folder_path: directory to save plots
        use_average: If True, plot moving average instead of raw scores
    
    Returns:
        list: Paths to saved plot files
    """
    os.makedirs(folder_path, exist_ok=True)
    plot_paths = []

    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].copy()
        
        if use_average:
            algo_df['rolling_avg'] = algo_df['score'].rolling(window=5, min_periods=1).mean()
            plot_data = algo_df['rolling_avg']
            ylabel = "Average Score (Window=5)"
        else:
            plot_data = algo_df['score']
            ylabel = "Score"

        plt.figure(figsize=(10, 6))
        plt.plot(algo_df['run'], plot_data, marker='o', color='blue')
        plt.title(f"Performance of {algo}", fontsize=16)
        plt.xlabel("Run Number", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True)

        plot_path = os.path.join(folder_path, f"{algo.replace(' ', '_')}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
    
    return plot_paths

def save_comparison_plot(df: pd.DataFrame, base_path: str, use_average: bool = False, file_name: str = None) -> str:
    """
    Save comparison plot of all algorithms
    
    Args:
        df: DataFrame with results
        base_path: Base directory to save plots
        use_average: If True, plot moving average instead of raw scores
    
    Returns:
        str: Path to saved comparison plot
    """
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(right=0.75)

    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].copy()
        
        if use_average:
            algo_df['plot_data'] = algo_df['score'].rolling(window=5, min_periods=1).mean()
            ylabel = "Average Score (Window=5)"
        else:
            algo_df['plot_data'] = algo_df['score']
            ylabel = "Score"
            
        plt.plot(algo_df['run'], algo_df['plot_data'], label=algo)

    plt.title("Algorithm Comparison", fontsize=16)
    plt.xlabel("Number of Runs", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.legend(title="Algorithms", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    if file_name is not None:
        combined_plot_path = os.path.join(base_path, f"{file_name}.png")
    else:
        combined_plot_path = os.path.join(base_path, "combined_plot.png")
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()

    return combined_plot_path

import ctypes

def get_screen_resolution():
    """Get the primary screen resolution"""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # width, height

def calculate_window_positions(num_windows):
    """Calculate positions for pygame windows in a grid layout"""
    screen_width, screen_height = get_screen_resolution()
    margin = 30 # Margin between windows and screen edges
    
    # Calculate optimal grid dimensions
    # Try different grid configurations to find best fit
    best_size = 0
    best_rows = 1
    best_cols = 1
    
    for rows in range(1, num_windows + 1):
        cols = (num_windows + rows - 1) // rows  # Ceiling division
        
        # Calculate maximum possible window size for this grid
        max_width = (screen_width - margin * (cols + 1)) // cols
        max_height = (screen_height - margin * (rows + 1) - 40 * rows) // rows # content top of pygame 40px height
        window_size = min(max_width, max_height)  # Keep square aspect ratio
        
        if window_size > best_size:
            best_size = window_size
            best_rows = rows
            best_cols = cols
    
    window_size = best_size
    positions = []
    
    # Calculate total grid width and height
    total_width = best_cols * window_size + (best_cols - 1) * margin
    total_height = best_rows * window_size + (best_rows - 1) * margin + 40 * best_rows
    
    # Calculate starting position to center the grid
    start_x = (screen_width - total_width) // 2
    start_y = (screen_height - total_height) // 2 + 40
    
    for i in range(num_windows):
        row = i // best_cols
        col = i % best_cols
        
        # Calculate position for each window
        x = int(start_x + col * (window_size + margin))
        y = int(start_y + row * (window_size + margin + 40))
        
        positions.append((x, y))
    
    return positions, window_size


def save_results(df, base_path="/content/drive/MyDrive/game_ai", folder_path:str="ComparedAgent", use_average = False):
    os.makedirs(base_path, exist_ok=True)
    if df.empty:
        return None, None, None
    
    individual_folder = os.path.join(base_path, folder_path)
    os.makedirs(individual_folder, exist_ok=True)

    csv_path = f"{individual_folder}/results_table.csv"
    df.to_csv(csv_path, index=False)

    # Generate and save plots
    individual_plots = save_individual_results(df=df, folder_path=individual_folder, use_average=use_average)
    comparison_plot = save_comparison_plot(df=df, base_path=base_path, use_average=use_average, file_name=folder_path)

    return csv_path, individual_plots, comparison_plot


if __name__ == "__main__":
    all_results = []
    algorithms = [
        # DodgeAlgorithm.DL_PARAM_BATCH_INTERVAL_NUMPY,
        # DodgeAlgorithm.DL_PARAM_LONG_SHORT_NUMPY,
        # DodgeAlgorithm.SUPERVISED, 
        DodgeAlgorithm.DL_VISION_LONG_SHORT_NUMPY,
        DodgeAlgorithm.DL_VISION_BATCH_INTERVAL_NUMPY,
        # DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        # DodgeAlgorithm.LEAST_DANGER_PATH,
        DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
        DodgeAlgorithm.RANDOM_SAFE_ZONE
    ]

    print(f"\n=== Benchmarking Deep Learning Bot ===")
    all_results = run_benchmark_parallel(algorithms, num_episodes=20)
    
    df = pd.DataFrame(all_results)
    
    # folder_path = "param_agent"
    # folder_path = "vision_agent"
    # folder_path = "heuristic_agent"
    folder_path = "test"
    save_results(df, base_path=base_path, folder_path=folder_path, use_average=USE_AVERAGE)