import numpy as np
from configs.bot_config import (
    DodgeAlgorithm, DODGE_ALGORITHM,
    SCAN_RADIUS, USE_COMPLEX_SCANNING, BOT_ACTION, BOT_DRAW)
from utils.draw_utils import draw_sector, draw_complex_sector
from bot.heuristic_dodge import HeuristicDodgeBot
from bot.deep_learning.param_input.numpy_agent import ParamNumpyAgent
from bot.deep_learning.param_input.pytorch_agent import ParamTorchAgent

class BotManager:
    def __init__(self, game):
        self.game = game
        self.current_bot = None
        
    def create_bot(self, algorithm: DodgeAlgorithm = DODGE_ALGORITHM, load_saved_model: bool = False):
        """Create a bot based on the specified dodge algorithm."""
        if algorithm in [
            DodgeAlgorithm.FURTHEST_SAFE_DIRECTION,
            DodgeAlgorithm.LEAST_DANGER_PATH,
            DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED,
            DodgeAlgorithm.RANDOM_SAFE_ZONE,
            DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION
        ]:
            self.current_bot = HeuristicDodgeBot(self.game, algorithm)
        else:
            if algorithm == DodgeAlgorithm.DL_PARAM_INPUT_NUMPY:
                self.current_bot = ParamNumpyAgent(self.game, load_saved_model)
            elif algorithm == DodgeAlgorithm.DL_PARAM_INPUT_TORCH:
                self.current_bot = ParamTorchAgent(self.game, load_saved_model)
        return self.current_bot
    
    def get_action(self, state) -> np.ndarray:
        """Get action from current bot"""
        if self.current_bot and BOT_ACTION:
            return self.current_bot.get_action(state)
        return None

    def draw_bot_vision(self):
        """Vẽ vision của bot và hướng di chuyển"""
        if not self.current_bot or not BOT_DRAW:
            return
            
        player = self.game.player
        surface = self.game.surface
        
        # Vẽ vòng tròn scan
        player.draw_surround_circle(SCAN_RADIUS)
        
        # Vẽ các sectors
        if USE_COMPLEX_SCANNING:
            self._draw_complex_sectors(SCAN_RADIUS)
        else:
            self._draw_simple_sectors(SCAN_RADIUS)
            
        # Tô màu đạn trong vùng scan    
        self.game.bullet_manager.color_in_radius(SCAN_RADIUS, (128,0,128))
        
        # Vẽ hướng di chuyển của bot
        bot_direction = self.get_action()
        if bot_direction:
            best_direction_index = np.argmax(bot_direction)
            if best_direction_index != 8:
                draw_sector(surface, player.x, player.y, 
                           50, best_direction_index, (0, 255, 0))

    def _draw_simple_sectors(self, radius: int):
        """Vẽ các sector đơn giản (chỉ chia theo góc)"""
        bullets_in_radius = self.game.bullet_manager.get_bullet_in_range(radius)
        sector_flags = self.game.bullet_manager.get_simple_regions(bullets_in_radius)
        
        for i, has_bullet in enumerate(sector_flags):
            if has_bullet:
                draw_sector(self.game.surface, self.game.player.x, 
                           self.game.player.y, radius, i, (255,255,0))

    def _draw_complex_sectors(self, radius: int, num_angle_divisions: int = 8, 
                            num_radius_divisions: int = 3):
        """Vẽ các sector phức tạp (chia theo cả góc và bán kính)"""
        bullets_in_radius = self.game.bullet_manager.get_bullet_in_range(radius)
        sector_flags = self.game.bullet_manager.get_complex_regions(
            bullets_in_radius, num_angle_divisions, num_radius_divisions)
            
        for i, has_bullet in enumerate(sector_flags):
            if has_bullet:
                draw_complex_sector(
                    self.game.surface, self.game.player.x, self.game.player.y,
                    i, radius, num_angle_divisions, num_radius_divisions, (255,255,0))