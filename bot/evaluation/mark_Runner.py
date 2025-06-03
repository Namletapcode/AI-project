import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

USE_COLAB = False

folder_path = 'saved_files/benchmark'
USE_AVERAGE = False

if USE_COLAB:
    project_root = '/content/AI-project'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    folder_path = "/content/drive/MyDrive/game_ai"
elif __name__ == "__main__":
    # only re-direct below if running this file
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_config import DodgeAlgorithm


def run_single_episode(algorithm, episode_index):
    game = Game()
    game.restart_game()
    bot_manager = BotManager(game)
    bot_manager.create_bot(algorithm, load_saved_model=True)

    if not bot_manager.is_heuristic:
        bot_manager.current_bot.set_mode("perform")
        bot_manager.current_bot.load_model()

    game_over = False

    while not game_over:
        state = game.get_state(bot_manager.is_heuristic, bot_manager.is_vision, bot_manager.method)
        action_idx = bot_manager.current_bot.get_action_idx(state)
        game.update(action_idx)
        game.draw()
        _, game_over = game.get_reward()

    score = game.score
    print(f"Episode {episode_index}, Score: {score}")

    return {
        "algorithm": algorithm.name,
        "run": episode_index + 1,
        "score": score,
    }


def run_benchmark_parallel(algorithm, num_episodes=20, num_workers=4):
    pool = multiprocessing.Pool(processes=num_workers)
    args = [(algorithm, i) for i in range(num_episodes)]
    results = pool.starmap(run_single_episode, args)
    pool.close()
    pool.join()
    return results

def save_individual_results(df: pd.DataFrame, base_path: str, use_average: bool = False) -> list:
    """
    Save individual algorithm performance plots
    
    Args:
        df: DataFrame with results
        base_path: Base directory to save plots
        use_average: If True, plot moving average instead of raw scores
    
    Returns:
        list: Paths to saved plot files
    """
    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)
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

        plot_path = os.path.join(plots_dir, f"{algo.replace(' ', '_')}_plot.png")
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
    plt.legend(title="Algorithms", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    if file_name is not None:
        combined_plot_path = os.path.join(base_path, f"{file_name}.png")
    else:
        combined_plot_path = os.path.join(base_path, "combined_plot.png")
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()

    return combined_plot_path

def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    os.makedirs(base_path, exist_ok=True)
    if df.empty:
        return None, None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    
    plots_dir = os.path.join(base_path, "individual_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate and save plots
    individual_plots = save_individual_results(df, base_path, USE_AVERAGE)
    comparison_plot = save_comparison_plot(df, base_path, USE_AVERAGE)

    return csv_path, individual_plots, comparison_plot


if __name__ == "__main__":
    all_results = []

    # heuristic_algorithms = [
    #     DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
    #     DodgeAlgorithm.LEAST_DANGER_PATH,
    #     DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
    #     DodgeAlgorithm.RANDOM_SAFE_ZONE,
    #     DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
    # ]

    # for alg in heuristic_algorithms:
    #     print(f"\n=== Benchmarking Heuristic Bot: {alg.name} ===")
    #     results = run_benchmark_parallel(alg, num_episodes=20, num_workers=4)
    #     all_results.extend(results)
    
    # df = pd.DataFrame(all_results)
    
    # save_comparison_plot(df, base_path=folder_path, use_average=False, file_name="Heuristic_combine")

    dl_algorithms = [
        DodgeAlgorithm.DL_PARAM_BATCH_INTERVAL_NUMPY,
        DodgeAlgorithm.DL_PARAM_LONG_SHORT_NUMPY,
        DodgeAlgorithm.SUPERVISED
    ]

    for alg in dl_algorithms:
        print(f"\n=== Benchmarking Deep Learning Bot: {alg.name} ===")
        results = run_benchmark_parallel(alg, num_episodes=20, num_workers=4)
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    save_comparison_plot(df, base_path=folder_path, use_average=False, file_name="ParamAndSupervised")
    
    # dl_algorithms = [
    #     DodgeAlgorithm.DL_VISION_BATCH_INTERVAL_NUMPY
    # ]

    # for alg in dl_algorithms:
    #     print(f"\n=== Benchmarking Deep Learning Bot: {alg.name} ===")
    #     results = run_benchmark_parallel(alg, num_episodes=20, num_workers=4)
    #     all_results.extend(results)
    
    # df = pd.DataFrame(all_results)
    # save_individual_results(df, base_path=folder_path, use_average=False)