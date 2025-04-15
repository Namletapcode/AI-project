import pygame
import math
import numpy as np
from collections import deque
from configs.game_config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BOX_SIZE, BOX_LEFT, BOX_TOP,
    PLAYER_SPEED, DISPLAY_PLAYER_TRAIL, TRAIL_MAX_LENGTH, UPDATE_DELTA_TIME)
from configs.bot_config import WALL_CLOSE_RANGE
from utils.draw_utils import draw_water_drop
    
class Player(pygame.sprite.Sprite):
    
    # optimize
    SQRT_2 = math.sqrt(2)
    
    def __init__(self, surface: pygame.Surface):
        super().__init__() #kế thừa lớp con từ lớp cha
        self.surface = surface
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
        self.color = (255, 0, 0)
        self.speed = PLAYER_SPEED
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.is_moving = False
        
        self.trail = deque(maxlen=TRAIL_MAX_LENGTH)  if DISPLAY_PLAYER_TRAIL else None
        
    def draw(self):
        if DISPLAY_PLAYER_TRAIL:
            self.trail.append((self.x, self.y))  # Lưu vị trí mới vào trail
            draw_water_drop(self.surface, self)
        pygame.draw.circle(self.surface, self.color, (self.x,self.y), self.radius)

    def draw_surround_circle(self, radius: float):
        pygame.draw.circle(self.surface, (255, 255, 255), (int(self.x), int(self.y)), radius, 1)
        
    def update(self, action: np.ndarray):
        self.move(action)
    
    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.direction = pygame.Vector2(0, 0)
        self.is_moving = False
        if DISPLAY_PLAYER_TRAIL:
            self.trail.clear()
        
    def set_movement_from_index(self, action: int):
        self.direction = self.directions[action]
    
    def input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            self.direction.x = -1
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            self.direction.x = 1
        else: self.direction.x = 0 
        if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            self.direction.y = -1
        elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
            self.direction.y = 1
        else: self.direction.y = 0
        
    def direction_to_position(self, direction: pygame.Vector2) -> pygame.Vector2:
        if direction.x and direction.y:
            x = self.x + direction.x * PLAYER_SPEED * UPDATE_DELTA_TIME / self.SQRT_2
            y = self.y + direction.y * PLAYER_SPEED * UPDATE_DELTA_TIME / self.SQRT_2
        else:
            x = self.x + direction.x * PLAYER_SPEED * UPDATE_DELTA_TIME
            y = self.y + direction.y * PLAYER_SPEED * UPDATE_DELTA_TIME
        
        x, y = self.handle_screen_collision(x, y)

        return pygame.Vector2(x, y)
        
    def move(self, action: np.ndarray = None):
        if action is None:
            # user keyboard input
            self.input()
        else:
            self.set_movement_from_index(np.argmax(action))
            
        if self.direction.x or self.direction.y:
            self.is_moving = True

        self.x, self.y = self.direction_to_position(self.direction)
        
    def handle_screen_collision(self, x, y):
        """Ngăn hình tròn đi ra ngoài màn hình"""
            
        left = BOX_LEFT
        top = BOX_TOP
        right = BOX_LEFT + BOX_SIZE
        bottom = BOX_TOP + BOX_SIZE

        if x - self.radius < left:
           x = left + self.radius  
        if x + self.radius > right:
           x = right - self.radius  
        if y - self.radius < top:
           y = top + self.radius  
        if y + self.radius > bottom:
           y = bottom - self.radius

        return x, y
    
    def get_near_wall_info(self):
        """
        Determines whether the player is near any of the four walls of the game area.

        The function checks the player's proximity to each wall within a predefined range
        (`WALL_CLOSE_RANGE`) and marks the corresponding direction as 1 if the player is close.

        Returns:
            list[int]: A list of four integers (0 or 1) representing proximity to walls:
        
        Index mapping:
            0: Near top box boarder
            1: Near right box boarder
            2: Near bottom box boarder
            3: Near left box boarder
        """
        result = [0] * 4

        if self.y - BOX_TOP < WALL_CLOSE_RANGE:
            result[0] = 1
        if BOX_LEFT + BOX_SIZE - self.x < WALL_CLOSE_RANGE:
            result[1] = 1
        if BOX_TOP + BOX_SIZE - self.y < WALL_CLOSE_RANGE:
            result[2] = 1
        if self.x - BOX_LEFT < WALL_CLOSE_RANGE:
            result[3] = 1

        return result