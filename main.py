from game.game_core import Game
from configs.bot_config import DodgeAlgorithm
from bot.bot_manager import BotManager

if __name__ == "__main__":
    game = Game()
    bot_manager = BotManager(game)
    bot = bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_TORCH)
    game.run(bot, bot_manager.draw_bot_vision)