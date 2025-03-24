import pygame
import math
import numpy as np
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED, BOX_SIZE, BOX_LEFT, BOX_TOP
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import Game
    
class Player(pygame.sprite.Sprite):
    
    # optimize
    SQRT_2 = math.sqrt(2)
    
    def __init__(self, game: "Game"):
        super().__init__() #kế thừa lớp con từ lớp cha
        self.game = game
        self.screen = game.screen
        self.screen_rect = game.screen_rect
        # self.rect = pygame.Rect(self.settings.screen_width//2,self.settings.screen_height-40,30,30)
        self.directions = [
            pygame.Vector2(1, 0),   # Phải
            pygame.Vector2(1, -1),  # Phải - Lên
            pygame.Vector2(0, -1),  # Lên
            pygame.Vector2(-1, -1), # Trái - Lên
            pygame.Vector2(-1, 0),  # Trái
            pygame.Vector2(-1, 1),  # Trái - Xuống
            pygame.Vector2(0, 1),   # Xuống
            pygame.Vector2(1, 1),    # Phải - Xuống
            pygame.Vector2(0, 0)   # Đứng yên
        ]
        self.direction = pygame.Vector2(0,0)
        self.radius = 5
        self.speed = PLAYER_SPEED
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        
    def draw(self):
        pygame.draw.circle(self.screen, (255,0,0), (self.x,self.y), self.radius)
        
    def update(self, action: np.ndarray = None):
        self.move(action)
        
    def set_movement_from_index(self, action: int):
        self.direction = self.directions[action]
    
    def input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.direction.x = -1
        elif keys[pygame.K_RIGHT]:
            self.direction.x = 1
        else: self.direction.x = 0 
        if keys[pygame.K_UP]:
            self.direction.y = -1
        elif keys[pygame.K_DOWN]:
            self.direction.y = 1
        else: self.direction.y = 0
        
    def direction_to_position(self, direction: pygame.Vector2) -> pygame.Vector2:
        if direction.x and direction.y:
            x = self.x + direction.x * PLAYER_SPEED / self.SQRT_2
            y = self.y + direction.y * PLAYER_SPEED / self.SQRT_2
        else:
            x = self.x + direction.x * PLAYER_SPEED
            y = self.y + direction.y * PLAYER_SPEED
        return pygame.Vector2(x, y)
        
    def move(self, action: np.ndarray = None):
        if action is None:
            # user keyboard input
            self.input()
        else:
            self.set_movement_from_index(np.argmax(action))
        if self.direction.x or self.direction.y:
            self.game.reward = 0
        self.x = self.direction_to_position(self.direction).x
        self.y = self.direction_to_position(self.direction).y
            
        self.handle_screen_collision()
        
    def handle_screen_collision(self):
        """Ngăn hình tròn đi ra ngoài màn hình"""
            
        left = BOX_LEFT
        top = BOX_TOP
        right = BOX_LEFT + BOX_SIZE
        bottom = BOX_TOP + BOX_SIZE

        if self.x - self.radius < left:
           self.x = left + self.radius  
        if self.x + self.radius > right:
           self.x = right - self.radius  
        if self.y - self.radius < top:
           self.y = top + self.radius  
        if self.y + self.radius > bottom:
           self.y = bottom - self.radius  