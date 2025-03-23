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
        self.direction = pygame.Vector2(0,0)
        self.bullets = pygame.sprite.Group()
        self.radius = 5
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        
    def draw_player(self):
        pygame.draw.circle(self.screen, (255,0,0), (self.x,self.y), self.radius)

    def update_player(self, action: np.ndarray = None):
        self.move(action)
    
    def move_left(self):
        self.direction.x = -1
    def move_right(self):
        self.direction.x = 1
    def move_up(self):
        self.direction.y = -1
    def move_down(self):
        self.direction.y = 1
    
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
  
    def move(self, action: np.ndarray = None):
        if not action:
            # user keyboard input
            self.input()
        else:
            # AI input
            if action[0] or action[4] or action[7]:
                self.direction.y = -1
            elif action[2] or action[5] or action[6]:
                self.direction.y = 1
            else:
                self.direction.y = 0
            if action[1] or action[4] or action[5]:
                self.direction.x = 1
            elif action[3] or action[6] or action[7]:
                self.direction.x = -1
            else:
                self.direction.x = 0

        if self.direction.x or self.direction.y:
            self.game.reward = 0

        if self.direction.x and self.direction.y:
            self.x += self.direction.x * PLAYER_SPEED / self.SQRT_2
            self.y += self.direction.y * PLAYER_SPEED / self.SQRT_2
        else:
            self.x += self.direction.x * PLAYER_SPEED
            self.y += self.direction.y * PLAYER_SPEED

        self.handle_screen_collision()

    def handle_screen_collision(self):
        """Ngăn hình tròn đi ra ngoài hộp"""

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