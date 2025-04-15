import pygame
import math
import random
from configs.game_config import BULLET_PATTERNS, SCREEN_HEIGHT, SCREEN_WIDTH, GAME_SPEED
from configs.bot_config import SCAN_RADIUS
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
    
    def update(self):
        self.bullets.update()

    def draw(self, surface):
        for bullet in self.bullets:
            bullet.draw(surface)
    
    def setup_timers(self):
        pygame.time.set_timer(pygame.USEREVENT + 1, int(BULLET_PATTERNS["ring"].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 2, int(BULLET_PATTERNS['rotating_ring'].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 3, int(BULLET_PATTERNS["spiral"].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 4, int(BULLET_PATTERNS["wave"].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 5, int(BULLET_PATTERNS["expanding_spiral"].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 6, int(BULLET_PATTERNS["bouncing"].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 7, int(BULLET_PATTERNS["spiral"].delay/GAME_SPEED))
        pygame.time.set_timer(pygame.USEREVENT + 8, int(BULLET_PATTERNS["ring"].delay/GAME_SPEED))
    
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
    
    def get_bullet_in_range(self, end_radius: int, start_radius: int = 0) -> list[Bullet]:
        """
        Retrieves a list of bullets that are within a specified distance range from the player.

        Args:
            end_radius (float): The maximum distance from the player within which bullets should be retrieved.
            start_radius (float, optional): The minimum distance from the player to consider bullets. Defaults to 0.

        Returns:
            list[Bullet]: A list of bullets that are within the specified range.
        """
        start_radius_square = start_radius * start_radius
        end_radius_square = end_radius * end_radius

        return [bullet for bullet in self.bullets 
            if start_radius_square <= (self.player.x - bullet.x) ** 2 + 
               (self.player.y - bullet.y) ** 2 <= end_radius_square]
    
    def get_complex_regions(self, bullets: list[Bullet], 
                        num_angle_divisions: int = 8, 
                        num_radius_divisions: int = 3) -> list[float]:
        """
        Converts bullet positions into a complex region representation based on both
        angle and distance from player.

        Args:
            bullets: List of bullets to analyze
            num_angle_divisions: Number of angular divisions (like 16 directions)
            num_radius_divisions: Number of radius divisions (rings around player)

        Returns:
            List[float]: List of length (num_angle_divisions * num_radius_divisions)
                        where 1 indicates bullet presence in that region
        """
        total_regions = num_angle_divisions * num_radius_divisions
        region_flags = [0] * total_regions
        
        sector_angle = 2 * math.pi / num_angle_divisions
        sector_radius = SCAN_RADIUS / num_radius_divisions
        start_angle = -sector_angle / 2
        
        for bullet in bullets:
            # Tính khoảng cách từ đạn đến player
            dx = bullet.x - self.player.x
            dy = bullet.y - self.player.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Xác định vòng (ring) chứa đạn
            ring_index = int(distance / sector_radius)
            
            # Tính góc của viên đạn
            angle = math.atan2(-dy, dx)
            # Chỉnh lại góc về phạm vi [0, 2*pi)
            angle = (angle - start_angle) % (2 * math.pi)
            
            # Xác định angle chứa đạn
            angle_index = int(angle // sector_angle)
            
            # Convert vị trí 2D (ring, angle) thành index 1D
            region_index = ring_index * num_angle_divisions + angle_index
            region_flags[region_index] = 1
            
        return region_flags
    
    def get_simple_regions(self, bullets: list[Bullet], num_sectors: int = 8) -> list[float]:
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

            # Chỉnh lại góc về phạm vi [0, 2*pi)
            angle = (angle - start_angle) % (2 * math.pi)

            # Xác định nan quạt nào chứa viên đạn
            sector_index = int(angle // sector_angle)
            sector_flags[sector_index] = 1

        return sector_flags