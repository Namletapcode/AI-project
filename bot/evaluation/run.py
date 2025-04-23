import os
import sys

project_root = '/content/AI-project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bot.evaluation.mark_Runner import BenchmarkRunner
from bot.heuristic_dodge import HeuristicDodgeBot
from game.game_core import Game
from configs.bot_config import DodgeAlgorithm
import csv



from enum import Enum

class DodgeAlgorithm(Enum):
    FURTHEST_SAFE_DIRECTION = 1
    LEAST_DANGER_PATH = 2
    LEAST_DANGER_PATH_ADVANCED = 3
    RANDOM_SAFE_ZONE = 4
    OPPOSITE_THREAT_DIRECTION = 5
def main():
    
    game = Game()
    dodgeMethod = {
    "Furthest Safe Direction": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.FURTHEST_SAFE_DIRECTION),
    "Least Danger": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.LEAST_DANGER_PATH),
    "Least Danger Advanced": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED),
    "Opposite Threat Direction": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION),
    "Random Safe Zone": lambda: HeuristicDodgeBot(game, DodgeAlgorithm.RANDOM_SAFE_ZONE),
}
    runner = BenchmarkRunner()
    save_path= "/content/drive/MyDrive/game_ai/benchmark_results.csv"
    runner.run(dodgeMethod,save_csv=True,csv_filename = save_path)
if __name__ == "__main__":
    main()
