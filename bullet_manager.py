import pygame
import math
import random
from settings import *
from bullet import Bullet

class BulletManager:
    def __init__(self):
        self.bullets = pygame.sprite.Group()
        self.spawn_time = 0
        self.angle_offset = 0
        self.radius = 5
    
    def get_random_corner(self):
        corners = [(0, 0), (SCREEN_WIDTH, 0), 
                   (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)]
        return random.choice(corners)
    
    def create_ring(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / RingBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, RingBullet().speed, color=RingBullet().color) 
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)  # Dùng add() với unpacking

    def create_spiral(self):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / SpiralBullet().num_bullets
        new_bullets = [Bullet(x, y, base_angle + i * angle_step, SpiralBullet().speed, fade=SpiralBullet().fade, color=SpiralBullet().color) 
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)  # Dùng add() với unpacking
        self.angle_offset += SpiralBullet().rotation_speed
    
    def create_targeted_shot(self, target_x, target_y, speed=4):
        x, y = self.get_random_corner()
        angle = math.atan2(target_y - y, target_x - x)
        self.bullets.add(Bullet(x, y, angle, speed))
    
    def create_rotating_ring(self):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / RotatingRingBullet().num_bullets
        new_bullets = [Bullet(x, y, base_angle + i * angle_step, RotatingRingBullet().speed, fade=RotatingRingBullet().fade, color=RotatingRingBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
        self.angle_offset += RotatingRingBullet().rotation_speed
    
    def create_wave(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / WaveBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, WaveBullet().speed, color=WaveBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
    
    def create_expanding_spiral(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / ExpandingSpiralBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, ExpandingSpiralBullet().initial_speed + i * ExpandingSpiralBullet().speed_increment, color=ExpandingSpiralBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
    
    def create_bouncing_bullets(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / BouncingBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, BouncingBullet().speed, color=BouncingBullet().color, bouncing=True)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
    
    def create_spiral_from_corners(self):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / SpiralBullet().num_bullets
        new_bullets = [Bullet(x, y, base_angle + i * angle_step, SpiralBullet().speed, color=SpiralBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
        self.angle_offset += SpiralBullet().rotation_speed
    
    def update(self):
        self.bullets.update()
        for bullet in self.bullets.copy():  # Lọc đạn ra ngoài màn hình
            if bullet.x < 0 or bullet.x > SCREEN_WIDTH or bullet.y < 0 or bullet.y > SCREEN_HEIGHT:
                self.bullets.remove(bullet)

    def draw(self, screen):
        # self.bullets.draw(screen)
        for bullet in self.bullets:
            bullet.draw(screen)