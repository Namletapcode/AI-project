"""import multiprocessing
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
    print(f"Bot mode: {bot_manager.current_bot.mode}")


    return {
        "algorithm": algorithm.name,
        "episode": episode_index,
        "score": score,
    }

def run_parallel_benchmark(algorithm, num_episodes=10):
    pool = multiprocessing.Pool(processes=num_episodes)
    args = [(algorithm, i) for i in range(num_episodes)]
    results = pool.starmap(run_single_episode, args)  # sửa hàm đúng tên
    pool.close()
    pool.join()
    return results

if __name__ == "__main__":
    dl_algorithms = [
        DodgeAlgorithm.DL_PARAM_INPUT_NUMPY,
        DodgeAlgorithm.DL_PARAM_INPUT_TORCH,
        DodgeAlgorithm.DL_VISION_INPUT_NUMPY,
    ]
    
    for alg in dl_algorithms:
        print(f"\n=== Benchmarking {alg.name} ===")
        results = run_parallel_benchmark(alg, num_episodes=10)
    
    csv_path = f"benchmark_{alg.name}.csv"
    keys = results[0].keys()
    with open(csv_path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"Đã lưu kết quả benchmark {alg.name} vào {csv_path}")"""
import multiprocessing
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
    bot_manager.create_bot(algorithm, load_saved_model=False)  

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
        "episode": episode_index,
        "score": score,
    }


def run_parallel_benchmark(algorithm, num_episodes=10):
    pool = multiprocessing.Pool(processes=num_episodes)
    args = [(algorithm, i) for i in range(num_episodes)]
    results = pool.starmap(run_single_episode, args)
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    heuristic_algorithms = [
        DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
        DodgeAlgorithm.LEAST_DANGER_PATH,
        DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
        DodgeAlgorithm.RANDOM_SAFE_ZONE,
        DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION,
    ]

    for alg in heuristic_algorithms:
        print(f"\n=== Benchmarking {alg.name} ===")
        results = run_parallel_benchmark(alg, num_episodes=10)

        if not results:
            print(f"Không có kết quả nào cho {alg.name}")
            continue

        csv_path = f"benchmark_{alg.name}.csv"
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        print(f" Đã lưu kết quả benchmark {alg.name} vào {csv_path}")

