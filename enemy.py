import pygame
import math
import random
from settings import Settings
from bullet_enemy import Bullet_enemy

class BulletManager:
    def __init__(self):
        self.bullets = []
        self.spawn_time = 0
        self.angle_offset = 0
        self.settings = Settings()
        self.radius = 5
    
    def get_random_corner(self):
        corners = [(0, 0), (self.settings.screen_width, 0), 
                   (0, self.settings.screen_height), (self.settings.screen_width, self.settings.screen_height)]
        return random.choice(corners)
    
    def create_ring(self, num_bullets=24, speed=3, fade=0):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))

    def create_spiral(self, num_bullets=36, speed=3, rotation_speed=5, fade=0):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))
        self.angle_offset += rotation_speed
    
    def create_targeted_shot(self, target_x, target_y, speed=4):
        x, y = self.get_random_corner()
        angle = math.atan2(target_y - y, target_x - x)
        self.bullets.append(Bullet_enemy(x, y, angle, speed))
    
    def create_rotating_ring(self, num_bullets=12, speed=3, rotation_speed=5, fade=0):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))
        self.angle_offset += rotation_speed
    
    def create_wave(self, num_bullets=10, speed=3, wave_amplitude=30):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed))
    
    def create_expanding_spiral(self, num_bullets=36, initial_speed=2, speed_increment=0.1):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = i * angle_step
            speed = initial_speed + i * speed_increment
            self.bullets.append(Bullet_enemy(x, y, angle, speed))
    
    def create_bouncing_bullets(self, num_bullets=10, speed=4):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, bouncing=True))
    
    def create_spiral_from_corners(self, screen_width, screen_height, num_bullets=36, speed=3, rotation_speed=5):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed))
        self.angle_offset += rotation_speed
    
    def update(self):
        for bullet in self.bullets:
            bullet.update()
        self.bullets = [b for b in self.bullets if 0 <= b.x <= self.settings.screen_width and 0 <= b.y <= self.settings.screen_height]

    def draw(self, screen):
        for bullet in self.bullets:
            bullet.draw(screen)
