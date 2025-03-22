import pygame
import math
from settings import SCREEN_WIDTH, SCREEN_HEIGHT

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, speed, fade=0, color=(255, 255, 255), bouncing=False, from_corner=False):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.fade = fade
        self.color = color
        self.bouncing = bouncing
        self.from_corner = from_corner
        self.radius = 5
        self.alpha = 255 if fade else None
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        self.image.fill((0, 0, 0, 0))
        pygame.draw.circle(self.image, self.color + (self.alpha if self.alpha else 255,), (self.radius, self.radius), self.radius)

    def update(self):
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
            self.image.fill((0, 0, 0, 0))
            pygame.draw.circle(self.image, self.color + (self.alpha,), (self.radius, self.radius), self.radius)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
    
    def set_color(self, color):
        self.color = color