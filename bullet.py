import pygame
import math
from collections import deque
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, USE_BULLET_COLORS,
                      DISPLAY_BULLET_TRAIL, TRAIL_MAX_LENGTH)
from help_methods import draw_water_drop

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, angle: int, speed: int, radius: int, color, fade=0, bouncing=False, from_corner=False):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.fade = fade
        self.origin_color = color if USE_BULLET_COLORS else (255, 255, 255)
        self.color = self.origin_color
        self.bouncing = bouncing
        self.from_corner = from_corner
        self.radius = radius
        self.alpha = 255 if fade else None
        
        self.trail = deque(maxlen=TRAIL_MAX_LENGTH) if DISPLAY_BULLET_TRAIL else None

    def update(self):
        # inefficient due to constantly re-calculating sine and cosine
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        if self.bouncing:
            if self.x - self.radius <= 0 or self.x + self.radius >= SCREEN_WIDTH:
                self.angle = math.pi - self.angle  
                self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
            if self.y - self.radius <= 0 or self.y + self.radius >= SCREEN_HEIGHT:
                self.angle = - self.angle  
                self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))

        if self.fade:
            self.alpha = max(0, self.alpha - self.fade)

    def draw(self, screen):
        if DISPLAY_BULLET_TRAIL:
            self.trail.append((self.x, self.y))
            draw_water_drop(screen, self)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
    
    def set_color(self, color):
        self.color = color