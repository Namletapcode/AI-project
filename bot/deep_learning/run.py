from mark_Runner import BenchmarkRunner
from bot.bot_manager import BotManager
from game.game_core import Game
from configs.bot_config import DodgeAlgorithm
import csv



from enum import Enum

class DodgeMethod(Enum):
    FURTHEST_SAFE_DIRECTION = 1
    LEAST_DANGER_PATH = 2
    LEAST_DANGER_PATH_ADVANCED = 3
    RANDOM_SAFE_ZONE = 4
    OPPOSITE_THREAT_DIRECTION = 5
def main():
    
    game = Game()
    dodgeMethod = {
    "Furthest Safe Direction": lambda: GameBot(game, DodgeMethod.FURTHEST_SAFE_DIRECTION),
    "Least Danger": lambda: GameBot(game, DodgeMethod.LEAST_DANGER_PATH),
    "Least Danger Advanced": lambda: GameBot(game, DodgeMethod.LEAST_DANGER_PATH_ADVANCED),
    "Opposite Threat Direction": lambda: GameBot(game, DodgeMethod.OPPOSITE_THREAT_DIRECTION),
    "Random Safe Zone": lambda: GameBot(game, DodgeMethod.RANDOM_SAFE_ZONE),
}
    runner = BenchmarkRunner()
    save_path= "/content/drive/MyDrive/game_ai/benchmark_results.csv"
    runner.run(dodgeMethod,save_csv=True,csv_filename = save_path)
if __name__ == "__main__":
    main()