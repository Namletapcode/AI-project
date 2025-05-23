import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import sys

project_root = '/content/AI-project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        state = game.get_state(bot_manager.is_heuristic, bot_manager.is_vision, bot_manager.is_numpy)
        action = bot_manager.current_bot.get_action(state)
        game.update(action)
        reward, game_over = game.get_reward()

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

def save_results(df, base_path="/content/drive/MyDrive/game_ai"):
    os.makedirs(base_path, exist_ok=True)
    if df.empty:
        return None, None

    csv_path = f"{base_path}/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    
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
    plt.legend(title="Algorithms", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    combined_plot_path = f"{base_path}/combined_plot.png"
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()

    return csv_path, plot_paths, combined_plot_path


if __name__ == "__main__":
    all_results = []

    heuristic_algorithms = [
        DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        DodgeAlgorithm.LEAST_DANGER_PATH,
        DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        DodgeAlgorithm.RANDOM_SAFE_ZONE,
        DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
    ]

    for alg in heuristic_algorithms:
        print(f"\n=== Benchmarking Heuristic Bot: {alg.name} ===")
        results = run_benchmark_parallel(alg, num_episodes=20, num_workers=4)
        all_results.extend(results)


    dl_algorithms = [
        DodgeAlgorithm.DL_PARAM_INPUT_NUMPY,
        DodgeAlgorithm.DL_PARAM_INPUT_TORCH,
        DodgeAlgorithm.DL_VISION_INPUT_NUMPY,
    ]

    for alg in dl_algorithms:
        print(f"\n=== Benchmarking Deep Learning Bot: {alg.name} ===")
        results = run_benchmark_parallel(alg, num_episodes=10, num_workers=4)
        all_results.extend(results)
        

    df = pd.DataFrame(all_results)
    save_results(df)

    print(" Đã lưu kết quả và biểu đồ vào Google Drive.")
