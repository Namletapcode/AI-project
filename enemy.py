import pygame
import math
from settings import Settings
from bullet_enemy import Bullet_enemy

class BulletManager:
    def __init__(self):
        self.bullets = []
        self.spawn_time = 0
        self.angle_offset = 0
        self.settings = Settings()
        self.radius = 5
    def create_ring(self, x, y, num_bullets=24, speed=3, fade=0):
        """Creates a circular bullet pattern"""
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))

    def create_spiral(self, x, y, num_bullets=36, speed=3, rotation_speed=5, fade=0):
        """Creates a spiral bullet pattern"""
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))
        self.angle_offset += rotation_speed

    def create_rotating_ring(self, x, y, num_bullets=12, speed=3, rotation_speed=5, fade=0):
        """Creates a rotating bullet ring"""
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))
        self.angle_offset += rotation_speed
    
    def create_wave(self, x, y, num_bullets=10, speed=3, wave_amplitude=30, fade=0):
        """Creates a wave pattern that oscillates"""
        for i in range(num_bullets):
            angle = math.radians(90) + math.sin((pygame.time.get_ticks() + i * 10) / 100) * wave_amplitude
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))

    def create_negative_speed_spiral(self, x, y, num_bullets=50, speed=-2, rotation_speed=5, fade=0, color=(255, 255, 255)):
        """Creates a spiral with bullets moving inwards (negative speed)"""
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade, color=color))
        self.angle_offset -= rotation_speed  # Rotate in opposite direction

    def create_expanding_spiral(self, x, y, num_bullets=36, initial_speed=2, speed_increment=0.1, fade=0, color=(255, 255, 255)):
        """Creates a spiral where bullets gradually speed up"""
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = base_angle + i * angle_step
            speed = initial_speed + (i * speed_increment)
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade, color=color))
        self.angle_offset += 5

    def create_bouncing_bullets(self, x, y, num_bullets=10, speed=4, fade=0):
        """Creates bullets that bounce off the screen edges"""
        angle_step = 2 * math.pi / num_bullets
        for i in range(num_bullets):
            angle = i * angle_step
            self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade, bouncing=True))

    def update(self):
        """Update bullet positions and handle bouncing logic"""
        for bullet in self.bullets:
            bullet.update()
            if hasattr(bullet, 'bouncing') and bullet.bouncing:
                if bullet.x <= 0 or bullet.x >= self.settings.screen_width:
                    bullet.angle = math.pi - bullet.angle  # Reflect X direction
                if bullet.y <= 0 or bullet.y >= self.settings.screen_height:
                    bullet.angle = -bullet.angle  # Reflect Y direction
        self.bullets = [b for b in self.bullets if 0 <= b.x <= self.settings.screen_width and 0 <= b.y <= self.settings.screen_height]

    def create_spiral_from_corners(self, screen_width, screen_height, num_bullets=36, speed=3, rotation_speed=5, fade=0):
        """Creates a spiral pattern from all four corners of the screen"""
        corners = [(0, 0), (screen_width, 0), (0, screen_height), (screen_width, screen_height)]
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / num_bullets
        for corner in corners:
            x, y = corner
            for i in range(num_bullets):
                angle = base_angle + i * angle_step
                self.bullets.append(Bullet_enemy(x, y, angle, speed, fade=fade))
        self.angle_offset += rotation_speed
    
    def update(self):
        """Update bullet positions and remove off-screen bullets"""
        for bullet in self.bullets:
            bullet.update()
        self.bullets = [b for b in self.bullets if 0 <= b.x <= self.settings.screen_width and 0 <= b.y <= self.settings.screen_height]

    def draw(self, screen):
        """Draw all bullets"""
        for bullet in self.bullets:
            bullet.draw(screen)
