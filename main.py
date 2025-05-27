from game.game_core import Game
from configs.game_config import SCREEN_HEIGHT,SCREEN_WIDTH
from configs.bot_config import DodgeAlgorithm
from menu import Menu
from options_menu import Options_Menu
import pygame
from bot.bot_manager import BotManager
import threading
import os
from configs.dynamic_config import launch_configs_window
import matplotlib.pyplot as plt

bot_type = DodgeAlgorithm.DL_PARAM_INPUT_NUMPY
game_render = True
bot_mode = "train"
show_graph = True

HEADLESS_MODE = False # For google colab
if HEADLESS_MODE:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    game_render = False
    show_graph = False

if __name__ == "__main__":
    # pygame.init()
    # screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # font = pygame.font.Font(None, 36)

    # Khởi tạo menu
    # menu = Menu(screen)
    # options_menu = Options_Menu(screen, font)
    # in_menu = False
    # in_options = False
    # control_mode = "AI"  # Mặc định
    # bullet_speed = 5  # Mặc định

    # Vòng lặp menu
    # while in_menu:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             exit()

    #         if in_options:
    #             # Xử lý menu tùy chọn
    #             result = options_menu.handle_input(event)
    #             if result == "Back":
    #                 in_options = False
    #                 menu.selected_index == 0
    #                 result = menu.handle_input(event)
    #             # Cập nhật các giá trị từ OptionsMenu
    #             control_mode = options_menu.control_mode
    #             bullet_speed = options_menu.bullet_speed

    #         else:
    #             # Xử lý menu chính
    #             result = menu.handle_input(event)
    #             if result == "Playing":
    #                 in_menu = False  # Bắt đầu game
    #             elif result == "Options":
    #                 in_options = True
    #             elif result == "Quit":
    #                 pygame.quit()
    #                 exit()

    #     # Vẽ menu
    #     if in_options:
    #         options_menu.draw()
    #     else:
    #         menu.draw()
    
    # gui_thread = threading.Thread(target = launch_configs_window, daemon = True) #Khởi tạo GUI
    # gui_thread.start()
    
    # Init plot before game for resolving resize window problem
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