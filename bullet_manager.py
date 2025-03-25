import pygame
import math
import random
import numpy as np
from settings import *
from bullet import Bullet
from player import Player

class BulletManager:
    def __init__(self, player: "Player"):
        self.bullets = pygame.sprite.Group()
        self.spawn_time = 0
        self.angle_offset = 0
        self.radius = 5
        self.player = player
    
    def get_state(self) -> np.ndarray:
        """
        Returns a numpy array of shape (12, 1), representing the player's surroundings.
    
        Format:
        - First 8 indices represent bullets in the 8 surrounding regions.
        - Last 4 indices represent proximity to the box boundaries.
    
        Index Mapping:
            0: Right
            1: Up-Right
            2: Up
            3: Up-Left
            4: Left
            5: Down-Left
            6: Down
            7: Down_right
            8: Near top of box
            9: Near right of box
            10: Near bottom of box
            11: Near left of box
        """
        result = np.zeros((12, ), dtype=np.float64)
        close_distance = 15
        
        close_bullets = self.get_bullet_in_range(close_distance)
        region_information = self.get_converted_regions(close_bullets)
        for i in range(len(region_information)):
            if region_information[i]:
                result[i] = 1
        # may extend into different parts with different distance ranges
        # example: 0 -> 25 pixels: part 1, 26 -> 30 pixels: part 2
        # therefore the result numpy.array might increase its size

        near_wall_info = self.player.get_near_wall_info()
        for i in range(len(near_wall_info)):
            if i == 1:
                result[i + 8] = 1

        return result.reshape(12, 1)

    def spawn_random_bullet_pattern(self):
        if pygame.time.get_ticks() % 100 == 0:
            self.create_ring()
        if pygame.time.get_ticks() % 151 == 0:
            self.create_rotating_ring()
        if pygame.time.get_ticks() % 199 == 0:
            self.create_spiral()
        if pygame.time.get_ticks() % 237 == 0:
            self.create_wave()
        if pygame.time.get_ticks() % 367 == 0:
            self.create_expanding_spiral()
        if pygame.time.get_ticks() % 403 == 0:
            self.create_bouncing_bullets()
        if pygame.time.get_ticks() % 450 == 0:
            self.create_spiral_from_corners()
        if pygame.time.get_ticks() % 100 == 0:
            self.create_targeted_shot(self.player.x, self.player.y, speed=DEFAULT_BULLET_SPEED)
    
    def get_random_corner(self) -> tuple[int, int]:
        corners = [(0, 0), (SCREEN_WIDTH, 0), 
                   (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)]
        return random.choice(corners)
    
    def create_ring(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / RingBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, RingBullet().speed, RingBullet().radius, color=RingBullet().color) 
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)  # Dùng add() với unpacking

    def create_spiral(self):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / SpiralBullet().num_bullets
        new_bullets = [Bullet(x, y, base_angle + i * angle_step, SpiralBullet().speed, SpiralBullet().radius, fade=SpiralBullet().fade, color=SpiralBullet().color) 
                       for i in range(SpiralBullet().num_bullets)]
        self.bullets.add(*new_bullets)
        self.angle_offset += SpiralBullet().rotation_speed
    
    def create_targeted_shot(self, target_x, target_y, speed=4):
        x, y = self.get_random_corner()
        angle = math.atan2(target_y - y, target_x - x)
        self.bullets.add(Bullet(x, y, angle, speed, RingBullet().radius, RingBullet().color))
    
    def create_rotating_ring(self):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / RotatingRingBullet().num_bullets
        new_bullets = [Bullet(x, y, base_angle + i * angle_step, RotatingRingBullet().speed, RotatingRingBullet().radius, fade=RotatingRingBullet().fade, color=RotatingRingBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
        self.angle_offset += RotatingRingBullet().rotation_speed
    
    def create_wave(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / WaveBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, WaveBullet().speed, WaveBullet().radius, color=WaveBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
    
    def create_expanding_spiral(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / ExpandingSpiralBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, ExpandingSpiralBullet().speed + i * ExpandingSpiralBullet().speed_increment, ExpandingSpiralBullet().radius, color=ExpandingSpiralBullet().color)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
    
    def create_bouncing_bullets(self):
        x, y = self.get_random_corner()
        angle_step = 2 * math.pi / BouncingBullet().num_bullets
        new_bullets = [Bullet(x, y, i * angle_step, BouncingBullet().speed, BouncingBullet().radius, color=BouncingBullet().color, bouncing=True)
                       for i in range(RingBullet().num_bullets)]
        self.bullets.add(*new_bullets)
    
    def create_spiral_from_corners(self):
        x, y = self.get_random_corner()
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / SpiralBullet().num_bullets
        new_bullets = [Bullet(x, y, base_angle + i * angle_step, SpiralBullet().speed, SpiralBullet().radius, color=SpiralBullet().color)
                       for i in range(SpiralBullet().num_bullets)]
        self.bullets.add(*new_bullets)
        self.angle_offset += SpiralBullet().rotation_speed
        
    def get_bullets_detail(self):
        return [(bullet.x, bullet.y, math.degrees(bullet.angle)) for bullet in self.bullets]
    
    def bullets_in_radius(self, screen: pygame.Surface, radius: int) -> tuple: # temporally not in use
        # pygame.draw.circle(screen, (255, 255, 255), (int(self.player.x), int(self.player.y)), radius, 1) moved into player.py
        
        bullets_in_radius = []
        radius = radius ** 2
        for bullet in self.bullets:
            # reduce sqrt calculation for more optimize
            distance = (self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2
            if distance <= radius:
                bullet.set_color((128, 0, 128))     # Đổi màu đạn thành tím
                bullets_in_radius.append(bullet)
            else:
                bullet.set_color(bullet.origin_color)   # Trở lại màu mặc định
        
        return bullets_in_radius
    
    def color_in_radius(self, radius = None, color = None):
        if not radius or not color:
            return
        radius_square = radius ** 2
        for bullet in self.bullets:
            distance_square = (self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2
            if distance_square <= radius_square:
                bullet.set_color(color)
            else:
                bullet.set_color(bullet.origin_color)
    
    def get_bullet_in_range(self, end_radius: float, start_radius: float = 0) -> list[Bullet]:
        """
        Retrieves a list of bullets that are within a specified distance range from the player.

        Args:
            end_radius (float): The maximum distance from the player within which bullets should be retrieved.
            start_radius (float, optional): The minimum distance from the player to consider bullets. Defaults to 0.

        Returns:
            list[Bullet]: A list of bullets that are within the specified range.
        """
        bullets = []
        start_radius_square: float = start_radius ** 2
        end_radius_square: float = end_radius ** 2

        for bullet in self.bullets:
            distance_square = (self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2
            if start_radius_square <= distance_square <= end_radius_square:
                bullets.append(bullet)

        return bullets
    
    def get_converted_regions(self, bullets: list[Bullet]) -> list[float]:
        """
        Converts bullet positions into an 8-region representation based on their angle 
        relative to the player.

        The function divides the space around the player into 8 directional regions, 
        assigning a value of 1 to any region that contains at least one bullet.

        Args:
            bullets (List[Bullet]): A list of bullets to be analyzed.

        Returns:
            List[float]: A list of 8 floats (either 0 or 1) representing whether bullets 
        
        Index Mapping:
            0: Right
            1: Up-Right
            2: Up
            3: Up-Left
            4: Left
            5: Down-Left
            6: Down
            7: Down_right
        """
        result = [0] * 8

        for bullet in bullets:
            angle = math.atan2(bullet.y - self.player.y, bullet.x - self.player.x)
            angle = math.degrees(angle)
            if -157.5 < angle <= -112.5:
                result[3] = 1
            elif -112.5 < angle <= -67.5:
                result[2] = 1
            elif -67.5 < angle <= -22.5:
                result[1] = 1
            elif -22.5 < angle <= 22.5:
                result[0] = 1
            elif 22.5 < angle <= 67.5:
                result[7] = 1
            elif 67.5 < angle <= 112.5:
                result[6] = 1
            elif 112.5 < angle <= 157.5:
                result[5] = 1
            else:
                result[4] = 1

        return result

    def update(self):
        self.spawn_random_bullet_pattern()
        self.bullets.update()
        for bullet in self.bullets.copy():  # Lọc đạn ra ngoài màn hình
            if bullet.x < 0 or bullet.x > SCREEN_WIDTH or bullet.y < 0 or bullet.y > SCREEN_HEIGHT:
                self.bullets.remove(bullet)

    def draw(self, screen):
        for bullet in self.bullets:
            bullet.draw(screen)