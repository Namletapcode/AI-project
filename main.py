from game.game_core import Game
from bot.heuristic_dodge import HeuristicDodgeBot
from configs.game_config import dt_max, FPS, UPS, UPDATE_DELTA_TIME
from configs.bot_config import DodgeMethod, BOT_DRAW
import pygame

if __name__ == "__main__":
    game = Game()
    bot = HeuristicDodgeBot(game, method=DodgeMethod.LEAST_DANGER_PATH)
    
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
            action = bot.get_action()
            game.update(action, delta_time=UPDATE_DELTA_TIME)
            update_time -= update_interval
            first_frame = False
            
        game.draw(draw_extra=bot.draw_vision if BOT_DRAW else None)