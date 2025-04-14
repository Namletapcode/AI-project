import pygame
import math
import random
import numpy as np
from configs.game_config import BULLET_PATTERNS, SCREEN_HEIGHT, SCREEN_WIDTH, GAME_SPEED
from configs.bot_config import WALL_CLOSE_RANGE
from game.bullet import Bullet
from game.player import Player

class BulletManager:
    def __init__(self, player: "Player"):
        self.bullets = pygame.sprite.Group()
        self.setup_timers()
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
        
        close_bullets = self.get_bullet_in_range(WALL_CLOSE_RANGE)
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
    
    def setup_timers(self):
        INT_GAME_SPEED = int(GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 1, BULLET_PATTERNS["ring"].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 2, BULLET_PATTERNS['rotating_ring'].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 3, BULLET_PATTERNS["spiral"].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 4, BULLET_PATTERNS["wave"].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 5, BULLET_PATTERNS["expanding_spiral"].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 6, BULLET_PATTERNS["bouncing"].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 7, BULLET_PATTERNS["spiral"].delay//INT_GAME_SPEED)
        pygame.time.set_timer(pygame.USEREVENT + 8, BULLET_PATTERNS["ring"].delay//INT_GAME_SPEED)
    
    def spawn_random_bullet_pattern(self, event):
        if event.type == pygame.USEREVENT + 1:
            self.create_ring()
        elif event.type == pygame.USEREVENT + 2:
            self.create_rotating_ring()
        elif event.type == pygame.USEREVENT + 3:
            self.create_spiral()
        elif event.type == pygame.USEREVENT + 4:
            self.create_wave()
        elif event.type == pygame.USEREVENT + 5:
            self.create_expanding_spiral()
        elif event.type == pygame.USEREVENT + 6:
            self.create_bouncing_bullets()
        elif event.type == pygame.USEREVENT + 7:
            self.create_targeted_shot(self.player.x, self.player.y)
    
    def get_random_corner(self) -> tuple[int, int]:
        corners = [(0, 0), (SCREEN_WIDTH, 0), 
                   (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)]
        return random.choice(corners)
    
    def create_ring(self):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["ring"]
        angle_step = 2 * math.pi / pattern.num_bullets
        new_bullets = [
            Bullet(x, y, i * angle_step, pattern.speed, pattern.radius, 
                  color=pattern.color, fade=pattern.fade) 
            for i in range(pattern.num_bullets)
        ]
        self.bullets.add(*new_bullets)  # Dùng add() với unpacking

    def create_spiral(self):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["spiral"]
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / pattern.num_bullets
        new_bullets = [
            Bullet(x, y, base_angle + i * angle_step, pattern.speed, 
                  pattern.radius, color=pattern.color, fade=pattern.fade)
            for i in range(pattern.num_bullets)
        ]
        self.bullets.add(*new_bullets)
        self.angle_offset += pattern.rotation_speed
    
    def create_targeted_shot(self, target_x, target_y):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["ring"]
        angle = math.atan2(target_y - y, target_x - x)
        self.bullets.add(Bullet(x, y, angle, pattern.speed, pattern.radius, pattern.color))
    
    def create_rotating_ring(self):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["rotating_ring"]
        base_angle = math.radians(self.angle_offset)
        angle_step = 2 * math.pi / pattern.num_bullets
        new_bullets = [
            Bullet(x, y, base_angle + i * angle_step, pattern.speed,
                  pattern.radius, color=pattern.color)
            for i in range(pattern.num_bullets)
        ]
        self.bullets.add(*new_bullets)
        self.angle_offset += pattern.rotation_speed
    
    def create_wave(self):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["wave"]
        angle_step = 2 * math.pi / pattern.num_bullets
        new_bullets = [
            Bullet(x, y, i * angle_step, pattern.speed, pattern.radius,
                  color=pattern.color, fade=pattern.fade)
            for i in range(pattern.num_bullets)
        ]
        self.bullets.add(*new_bullets)
    
    def create_expanding_spiral(self):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["expanding_spiral"]
        angle_step = 2 * math.pi / pattern.num_bullets
        new_bullets = [
            Bullet(x, y, i * angle_step, 
                  pattern.speed + i * pattern.speed_increment,
                  pattern.radius, color=pattern.color)
            for i in range(pattern.num_bullets)
        ]
        self.bullets.add(*new_bullets)
    
    def create_bouncing_bullets(self):
        x, y = self.get_random_corner()
        pattern = BULLET_PATTERNS["bouncing"]
        angle_step = 2 * math.pi / pattern.num_bullets
        new_bullets = [
            Bullet(x, y, i * angle_step, pattern.speed, pattern.radius,
                  color=pattern.color, bouncing=True)
            for i in range(pattern.num_bullets)
        ]
        self.bullets.add(*new_bullets)
        
    def get_bullets_detail(self):
        return [(bullet.x, bullet.y, math.degrees(bullet.angle)) for bullet in self.bullets]
    
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
        start_radius_square: float = start_radius ** 2
        end_radius_square: float = end_radius ** 2

        return [bullet for bullet in self.bullets 
            if start_radius_square <= (self.player.x - bullet.x) ** 2 + 
               (self.player.y - bullet.y) ** 2 <= end_radius_square]
    
    def get_converted_regions(self, bullets: list[Bullet], num_sectors: int = 8) -> list[float]:
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
        sector_flags = [0] * num_sectors
        sector_angle = 2 * math.pi / num_sectors  # Góc mỗi nan quạt
        start_angle = -sector_angle / 2

        for bullet in bullets:
            # Tính góc của viên đạn so với nhân vật
            angle = math.atan2(self.player.y - bullet.y, bullet.x - self.player.x)

            # Chỉnh lại góc về phạm vi [0, 360)
            angle = (angle - start_angle) % (2 * math.pi)

            # Xác định nan quạt nào chứa viên đạn
            sector_index = int(angle // sector_angle)
            sector_flags[sector_index] = 1

        return sector_flags

    def update(self, delta_time: float = 0.1/60000):
        self.bullets.update(delta_time)

    def draw(self, screen):
        for bullet in self.bullets:
            bullet.draw(screen)