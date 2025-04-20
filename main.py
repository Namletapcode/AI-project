from game.game_core import Game
from configs.game_config import dt_max, FPS, UPS
from configs.bot_config import DodgeAlgorithm
from bot.bot_manager import BotManager
import pygame

if __name__ == "__main__":
    game = Game()
    bot_manager = BotManager(game)
    bot_manager.create_bot(DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED)
    
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
            action = bot_manager.get_action()
            game.update(action)
            update_time -= update_interval
            first_frame = False
            
        game.draw(draw_extra=bot_manager.draw_bot_vision)
        # print("-------------")
        # for i, state in enumerate(game.get_state()):
        #     if state == 1: print(i)
        # print("-------------")