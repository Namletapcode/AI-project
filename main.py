from game.game_core import Game
from bot.heuristic_dodge import HeuristicDodgeBot
from configs.game_config import dt_max, FPS, UPS, UPDATE_DELTA_TIME,SCREEN_HEIGHT,SCREEN_WIDTH
from configs.bot_config import DodgeAlgorithm, BOT_DRAW
from menu import Menu
from options_menu import Options_Menu
import pygame
from bot.bot_manager import BotManager
import threading
from configs.dynamic_config import launch_configs_window

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 36)

    # Khởi tạo menu
    menu = Menu(screen)
    options_menu = Options_Menu(screen, font)
    in_menu = True
    in_options = False
    control_mode = "AI"  # Mặc định
    bullet_speed = 5  # Mặc định

    # Vòng lặp menu
    while in_menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if in_options:
                # Xử lý menu tùy chọn
                result = options_menu.handle_input(event)
                if result == "Back":
                    in_options = False
                    menu.selected_index == 0
                    result = menu.handle_input(event)
                # Cập nhật các giá trị từ OptionsMenu
                control_mode = options_menu.control_mode
                bullet_speed = options_menu.bullet_speed

            else:
                # Xử lý menu chính
                result = menu.handle_input(event)
                if result == "Playing":
                    in_menu = False  # Bắt đầu game
                elif result == "Options":
                    in_options = True
                elif result == "Quit":
                    pygame.quit()
                    exit()

        # Vẽ menu
        if in_options:
            options_menu.draw()
        else:
            menu.draw()
    
    gui_thread = threading.Thread(target = launch_configs_window, daemon = True) #Khởi tạo GUI
    gui_thread.start()
    game = Game()
    bot_manager = BotManager(game)
    bot_manager.create_bot(DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED)
    
    bot = bot_manager.create_bot(DodgeAlgorithm.DL_PARAM_INPUT_NUMPY)
    game.run(bot, mode="train", render=True, draw_extra=bot_manager.draw_bot_vision)
    clock = pygame.time.Clock()
    
    update_time = 0
    # Thời gian giữa các lần cập nhật game (seconds) (giảm khi GAME_SPEED tăng)
    update_interval = 1.0 / UPS
    
    first_frame = True
    
    while True:
        frame_time = min(clock.tick(FPS) / 1000, dt_max)
        
        update_time += frame_time
        
        # dùng first_frame để cập nhật ngay frame đầu tiên (tránh việc không update được trước khi draw)
        while update_time >= update_interval or first_frame:
            if options_menu.control_mode == "Player":
                action = None
            else: action = bot_manager.get_action()
            # action = bot_manager.get_action()
            game.update(action)
            update_time -= update_interval
            first_frame = False
            
        game.draw(draw_extra=bot_manager.draw_bot_vision)
        # print("-------------")
        # for i, state in enumerate(game.get_state()):
        #     if state == 1: print(i)
        # print("-------------")
