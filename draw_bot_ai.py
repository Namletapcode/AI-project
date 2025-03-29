
import numpy as np
from settings import DodgeMethod,DrawSectorMethod, BOT_ACTION, FILTER_MOVE_INTO_WALL, SCAN_RADIUS
from help_methods import draw_sector

class Draw_bot_ai:
    def __init__(self,bot_ai):
        self.bot_ai=bot_ai
        self.player=bot_ai.player
        self.game=bot_ai.game
        self.screen=bot_ai.screen
    
    def draw_sectors(self, radius, num_sectors=None):
        if num_sectors is None:
            # Lấy danh sách đạn trong bán kính d
            bullets_in_radius = self.game.bullet_manager.get_bullet_in_range(radius)
            
            # Phân loại đạn vào các nan quạt
            sector_flags = self.game.bullet_manager.get_converted_regions(bullets_in_radius)

            num_sectors = len(sector_flags)

            for i in range(num_sectors):
                # Chọn màu: Vàng nếu có đạn, Trắng nếu không
                color = (255, 255, 0) if sector_flags[i] else (255, 255, 255)
                
                # Vẽ viền cung tròn
                if sector_flags[i]:
                    draw_sector(self.screen, self.player.x, self.player.y, radius, i, color, num_sectors)
                    
    def draw_vison(self):
        self.player.draw_surround_circle(SCAN_RADIUS)
        self.draw_sectors(SCAN_RADIUS, None)
        self.game.bullet_manager.color_in_radius(SCAN_RADIUS, (128, 0, 128))
        best_direction_index = np.argmax(self.action)
        if best_direction_index != 8:
            draw_sector(self.screen, self.player.x, self.player.y, 50, best_direction_index, (0, 255, 0))
    

    