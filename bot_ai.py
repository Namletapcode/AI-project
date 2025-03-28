
import numpy as np
from typing import TYPE_CHECKING
from draw_bot_ai import Draw_bot_ai
from update_bot_ai import Update_bot_ai
from settings import DodgeMethod, BOT_ACTION

if TYPE_CHECKING:
    from game import Game

class GameBot:
    def __init__(self, game: "Game", method = DodgeMethod.FURTHEST_SAFE_DIRECTION):
        self.method = method
        self.game = game
        self.player = self.game.player
        self.bullets = self.game.bullet_manager.bullets
        self.screen = self.game.screen
        self.action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  # if self.is_activate() else None
        self.draw = Draw_bot_ai(self)
        self.update = Update_bot_ai(self)
        
    
    
