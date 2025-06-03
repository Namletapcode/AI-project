import os, sys
import matplotlib.pyplot as plt
from game.game_core import Game
from configs.bot_config import DodgeAlgorithm, SharedState
from bot.bot_manager import BotManager
import threading

bot_type = DodgeAlgorithm.DL_VISION_BATCH_INTERVAL_NUMPY
game_render = True
bot_mode = "train"
show_graph = True

HEADLESS_MODE = False # For google colab

if os.getenv("COLAB_RELEASE_TAG"):
    HEADLESS_MODE = True
    
if HEADLESS_MODE:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    game_render = False
    show_graph = False
if bot_mode == "perform":
    show_graph = False

def run_settings(share_state):
    from PySide6.QtWidgets import QApplication
    from utils.interface import SettingsWindow
    app = QApplication(sys.argv)
    window = SettingsWindow(share_state)
    window.move(1100, 200)
    window.show()
    app.exec()

class GameThread(threading.Thread):
    def __init__(self, share_state):
        super().__init__()
        self.share_state = share_state

    def run(self):
        game = Game(self.share_state)
        bot_manager = BotManager(game)
        
        bot_manager.create_bot(bot_type, True)
        game.run(bot_manager, mode=bot_mode, render=game_render, show_graph=show_graph)
    
if __name__ == "__main__":
    if bot_mode == "perform":
        share_state = SharedState()
        
        game_thread = GameThread(share_state)
        game_thread.start()
        
        run_settings(share_state)
            
        game_thread.join()
    elif bot_mode == "train":
        if not HEADLESS_MODE:
            if show_graph:
                plt.ion()
                os.environ['SDL_VIDEO_WINDOW_POS'] = '200,280' # Move pygame window
            else:
                plt.ioff()
            manager = plt.get_current_fig_manager()
            manager.window.move(690, 200) # Move plot window
        game = Game()
        bot_manager = BotManager(game)
        
        bot_manager.create_bot(bot_type)
        game.run(bot_manager, mode=bot_mode, render=game_render, show_graph=show_graph)