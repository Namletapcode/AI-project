import pygame
import math
import random
from configs.game_config import BULLET_PATTERNS, SCREEN_HEIGHT, SCREEN_WIDTH, GAME_SPEED, DEFAULT_BULLET_SPEED
from configs.bot_config import SCAN_RADIUS
from game.bullet import Bullet
from game.player import Player

class BulletManager:
    def __init__(self, player: "Player"):
        self.player = player
        self.key = 0
        self.reset(0)

    def get_random_point(self) -> tuple[int, int]:
        """padding = 40
        corners = [(padding, padding), (SCREEN_WIDTH - padding, padding), (padding, SCREEN_HEIGHT - padding), (SCREEN_WIDTH - padding, SCREEN_HEIGHT - padding),
                   (padding, SCREEN_HEIGHT / 2), (SCREEN_WIDTH - padding, SCREEN_HEIGHT / 2), (SCREEN_WIDTH / 2, padding), (SCREEN_WIDTH / 2, SCREEN_HEIGHT - padding)]"""
        padding = 0
        corners = [(padding, padding), (SCREEN_WIDTH - padding, padding), (padding, SCREEN_HEIGHT - padding), (SCREEN_WIDTH - padding, SCREEN_HEIGHT - padding),]
        return random.choice(corners)
    
    def random_delay(self, event) -> int:
        return random.randint(-event["prop"].delay_offset_limit, event["prop"].delay_offset_limit)
    
    def create_bullet_type(self, x, y, event: dict, update_num: int):
        if event["type"] == "spiral":
            if random.random() < event["prop"].probability or event["spawning"]:
                event["spawning"] = True
                if event["spawned"] == 0:
                    event["spawn_x"], event["spawn_y"] = x, y
                if update_num >= event["spawn_time"] + event["prop"].interval_delay * event["spawned"]:
                    base_angle = math.radians(self.angle_offset)
                    angle_step = 2 * math.pi / event["prop"].num_bullets
                    self.bullets.add(Bullet(event["spawn_x"], event["spawn_y"], base_angle + event["spawned"] * angle_step, event["prop"].speed, event["prop"].radius, color=event["prop"].color))
                    self.angle_offset += event["prop"].rotation_speed
                    event["spawned"] += 1
                if event["spawned"] >= event["prop"].num_bullets:
                    event["spawned"] = 0
                    event["spawn_time"] = update_num + event["prop"].delay + self.random_delay(event)
                    self.angle_offset = 0
                    event["spawning"] = False
        
        elif event["type"] == "tornado":
            if random.random() < event["prop"].probability or event["spawning"]:
                event["spawning"] = True
                if event["spawned"] == 0:
                    event["spawn_x"], event["spawn_y"] = SCREEN_HEIGHT/2, SCREEN_WIDTH/2
                if update_num >= event["spawn_time"] + event["prop"].interval_delay * event["spawned"]:
                    base_angle = math.radians(self.angle_offset)
                    angle_step = 2 * math.pi / event["prop"].num_bullets
                    Bullets = [Bullet(event["spawn_x"], event["spawn_y"], (2*math.pi/6)*i + base_angle + event["spawned"] * angle_step, event["prop"].speed, event["prop"].radius, color=event["prop"].color) for i in range(event["prop"].num_bullets)]
                    self.bullets.add(Bullets)
                    self.angle_offset += event["prop"].rotation_speed
                    event["spawned"] += 1
                if event["spawned"] >= event["prop"].num_bullets:
                    event["spawned"] = 0
                    event["spawn_time"] = update_num + event["prop"].delay + self.random_delay(event)
                    self.angle_offset = 0
                    event["spawning"] = False
        
        elif event["type"] == "sin_wave":
            if random.random() < event["prop"].probability or event["spawning"]:
                event["spawning"] = True

                if event["spawned"] == 0:
                    event["spawn_x"], event["spawn_y"] = x, y
                    event["base_x"], event["base_y"] = x, y
                    event["angle"] = math.atan2(self.player.y - event["spawn_y"], self.player.x - event["spawn_x"])

                if update_num >= event["spawn_time"] + event["prop"].interval_delay * event["spawned"]:
                    # Góc vuông góc với góc bắn
                    perpendicular_angle = event["angle"] + math.pi / 2

                    # Wave offset dao động theo 1 chu kỳ sin
                    wave_phase = 4 * math.pi * (event["spawned"] / event["prop"].count)
                    wave_offset = math.sin(wave_phase) * 30  # 30 là biên độ

                    # Lệch theo hướng vuông góc
                    spawn_x = event["base_x"] + wave_offset * math.cos(perpendicular_angle)
                    spawn_y = event["base_y"] + wave_offset * math.sin(perpendicular_angle)

                    self.bullets.add(Bullet(spawn_x, spawn_y, event["angle"], event["prop"].speed, event["prop"].radius, color=event["prop"].color))

                    event["spawned"] += 1

                if event["spawned"] >= event["prop"].count:
                    event["spawned"] = 0
                    event["spawn_time"] = update_num + event["prop"].delay + self.random_delay(event)
                    event["spawning"] = False

        elif event["type"] == "ring":
            if random.random() < event["prop"].probability:
                event["spawn_x"], event["spawn_y"] = x, y
                angle_step = 2 * math.pi / event["prop"].num_bullets
                new_bullets = [Bullet(event["spawn_x"], event["spawn_y"], i * angle_step, event["prop"].speed, event["prop"].radius, color=event["prop"].color) 
                        for i in range(event["prop"].num_bullets)]
                self.bullets.add(*new_bullets)
                event["spawn_time"] = update_num + event["prop"].delay + self.random_delay(event)

        elif event["type"] == "targeted_shot":
            if random.random() < event["prop"].probability:
                event["spawn_x"], event["spawn_y"] = x, y
                angle = math.atan2(self.player.x - y, self.player.y - x)
                self.bullets.add(Bullet(event["spawn_x"], event["spawn_y"], angle, DEFAULT_BULLET_SPEED, event["prop"].radius, event["prop"].color))
                event["spawn_time"] = update_num + event["prop"].delay + self.random_delay(event)

        elif event["type"] == "bouncing":
            if random.random() < event["prop"].probability:
                event["spawn_x"], event["spawn_y"] = x, y
                angle_step = 2 * math.pi / event["prop"].num_bullets
                new_bullets = [Bullet(event["spawn_x"], event["spawn_y"], i * angle_step, event["prop"].speed, event["prop"].radius, color=event["prop"].color, bouncing=True)
                            for i in range(event["prop"].num_bullets)]
                self.bullets.add(*new_bullets)
                event["spawn_time"] = update_num + event["prop"].delay + self.random_delay(event)
        
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
            num_angle_divisions: Number of angular divisions (like 8 directions)
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
            if distance >= SCAN_RADIUS:
                continue
            
            # Xác định vòng (ring) chứa đạn
            ring_index = (num_radius_divisions - 1) if distance >= SCAN_RADIUS else int(distance / sector_radius)
            
            # Tính góc của viên đạn
            angle = math.atan2(-dy, dx)
            
            # Dịch trục quay trái π/8 để vùng 0 nằm giữa start_angle và start_angle + sector_angle
            angle_shifted = (angle - start_angle) % (2 * math.pi)
            
            # Tính index vùng (mỗi vùng rộng π/4), đảm bảo kết quả trong [0, 7]
            angle_index = int(angle_shifted // sector_angle) % 8
            
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

    def update(self, update_num: int):
        for event in self.spawn_event:
            if update_num >= event["spawn_time"] and event["prop"].enable:
                self.create_bullet_type(*self.get_random_point(), event, update_num)
        self.bullets.update()

    def draw(self, screen):
        for bullet in self.bullets:
            bullet.draw(screen)

    def reset(self, update_count: int = 0):
        self.bullets = pygame.sprite.Group()
        self.spawn_time = 0
        self.angle_offset = 0
        self.radius = 5
        self.spawn_event = [
            {"type": "ring",     "spawn_time": BULLET_PATTERNS["ring"].delay + update_count,     "spawned": 0, "spawn_x": 0, "spawn_y": 0, "prop": BULLET_PATTERNS["ring"]},
            {"type": "target",   "spawn_time": BULLET_PATTERNS["ring"].delay + update_count,     "spawned": 0, "spawn_x": 0, "spawn_y": 0, "prop": BULLET_PATTERNS["ring"]},
            {"type": "bouncing", "spawn_time": BULLET_PATTERNS["bouncing"].delay + update_count, "spawned": 0, "spawn_x": 0, "spawn_y": 0, "prop": BULLET_PATTERNS["bouncing"]},
            {"type": "spiral",   "spawn_time": BULLET_PATTERNS["spiral"].delay + update_count,   "spawned": 0, "spawn_x": 0, "spawn_y": 0, "prop": BULLET_PATTERNS["spiral"], "spawning": False},
            {"type": "tornado",  "spawn_time": BULLET_PATTERNS["tornado"].delay + update_count,  "spawned": 0, "spawn_x": 0, "spawn_y": 0, "prop": BULLET_PATTERNS["tornado"], "spawning": False},
            {"type": "sin_wave", "spawn_time": BULLET_PATTERNS["sin_wave"].delay + update_count, "spawned": 0, "spawn_x": 0, "spawn_y": 0, "prop": BULLET_PATTERNS["sin_wave"], "base_x": 0, "base_y": 0, "angle": 0, "spawning": False},
        ]
