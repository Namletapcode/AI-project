import multiprocessing
import pandas as pd
import os
import sys 

import sys
project_root = '/content/AI-project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game.game_core import Game
from bot.bot_manager import BotManager
from configs.bot_configs import DodgeAlgorithm


def run_singel_episode(algorithm,episode_index):
    game= Game()
    bot_manager = BotManager(game)
    bot_manager.creat_bot(algorithm, load_saved_model =True)

    game_over = False 


    while not game_over:
        state = game.get_state(bot_manager.is_heuristic, bot_manager.is_vision, bot_manager.is_numpy)
        action = bot_manager.current_bot.get_action(state)
        game.update(action)
        reward, game_over = game.get_reward()
    
    score = game.score 

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
    alg = DodgeAlgorithm.DL_PARAM_INPUT_NUMPY  
    results = run_parallel_benchmark(alg, num_episodes=10)
    
    csv_path = f"benchmark_{alg.name}.csv"
    keys = results[0].keys()
    with open(csv_path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"Đã lưu kết quả benchmark {alg.name} vào {csv_path}")
