import pygame
import math
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import Game
    
class Player(pygame.sprite.Sprite):
    def __init__(self, game: "Game"):
        super().__init__() #kế thừa lớp con từ lớp cha
        self.screen = game.screen
        self.screen_rect = game.screen_rect
        # self.rect = pygame.Rect(self.settings.screen_width//2,self.settings.screen_height-40,30,30)
        self.direction = pygame.Vector2(0,0)
        self.bullets = pygame.sprite.Group()
        self.radius = 5
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 100
        
    def draw_player(self):
        pygame.draw.circle(self.screen, (255,0,0), (self.x,self.y), self.radius)

    def update_player(self):
        self.input()
        self.move()
    
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
  
    def move(self):
        if self.direction.x and self.direction.y:
            self.x += self.direction.x * PLAYER_SPEED / math.sqrt(2)
            self.y += self.direction.y * PLAYER_SPEED / math.sqrt(2)
        else:
            self.x += self.direction.x * PLAYER_SPEED
            self.y += self.direction.y * PLAYER_SPEED

    def handle_screen_collision(self):
        """Ngăn hình tròn đi ra ngoài màn hình"""
        if self.x - self.radius < 0:
            self.x = self.radius  # Giữ trong giới hạn trái
        if self.x + self.radius > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH - self.radius  # Giữ trong giới hạn phải
        if self.y - self.radius < 0:
            self.y = self.radius  # Giữ trong giới hạn trên
        if self.y + self.radius > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.radius
        box_x, box_y, box_size = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 500
        left = box_x - box_size // 2
        right = box_x + box_size // 2
        top = box_y - box_size // 2
        bottom = box_y + box_size // 2

        if self.x - self.radius < left:
           self.x = left + self.radius  
        if self.x + self.radius > right:
           self.x = right - self.radius  
        if self.y - self.radius < top:
           self.y = top + self.radius  
        if self.y + self.radius > bottom:
           self.y = bottom - self.radius  