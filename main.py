import os, sys, pygame
import matplotlib.pyplot as plt
from game.game_core import Game
from configs.bot_config import DodgeAlgorithm
from bot.bot_manager import BotManager
from PySide6.QtWidgets import QApplication
from utils.interface import SettingsWindow

bot_type = DodgeAlgorithm.DL_VISION_INPUT_NUMPY
game_render = True
bot_mode = "train"
show_graph = True

HEADLESS_MODE = False # For google colab
if HEADLESS_MODE:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    game_render = False
    show_graph = False
if bot_mode == "perform":
    show_graph = False

def run_settings():
    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.move(1100, 200)
    window.show()
    app.exec()

def run_game():
    if not HEADLESS_MODE:
        if show_graph:
            plt.ion()
            os.environ['SDL_VIDEO_WINDOW_POS'] = '200,280' # Move pygame window
        else:
            plt.ioff()
        plt.figure()
        manager = plt.get_current_fig_manager()
        manager.window.move(690, 200) # Move plot window
    
    game = Game()
    bot_manager = BotManager(game)
    
    bot_manager.create_bot(bot_type)
    game.run(bot_manager, mode=bot_mode, render=game_render, show_graph=show_graph)
    
if __name__ == "__main__":
    if bot_mode == "perform":
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)  # Use spawn method for multiprocessing
        
        settings_process = mp.Process(target=run_settings)
        settings_process.start()
        # Run game in main thread
        run_game()
        
        settings_process.join()  # Wait for settings window to close before exiting
    else:
        run_game()