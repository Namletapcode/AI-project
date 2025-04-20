import pygame
import math
import random
import numpy as np
from configs.settings import *
from bullet import Bullet
from player import Player
import time

class BulletManager:
    def __init__(self, player: "Player"):
        self.bullets = pygame.sprite.Group()
        self.bullets.add(Bullet(BOX_LEFT, BOX_TOP, 0, 0, 0, RED))
        self.bullets.add(Bullet(BOX_LEFT + BOX_SIZE, BOX_TOP , 0, 0, 0, RED))
        self.bullets.add(Bullet(BOX_LEFT, BOX_TOP + BOX_SIZE, 0, 0, 0, RED))
        self.bullets.add(Bullet(BOX_LEFT + BOX_SIZE, BOX_TOP + BOX_SIZE, 0, 0, 0, RED))
        self.spawn_time = 0
        self.angle_offset = 0
        self.radius = 5
        self.player = player
        self.spawn_event = [
            {"bullet_type": "ring",             "spawn_time":0, "init_delay": RingBullet().delay,            "delay": 0, "count": RingBullet().num_bullets},
            {"bullet_type": "spiral",           "spawn_time":0, "init_delay": SpiralBullet().delay,          "delay": 100, "count": SpiralBullet().num_bullets, "spawned": 0},
            {"bullet_type": "targeted_shot",    "spawn_time":0, "init_delay": 2000,                          "delay": 0, "count": 5},    
            {"bullet_type": "rotating_ring",    "spawn_time":0, "init_delay": RotatingRingBullet().delay,    "delay": 0, "count": RotatingRingBullet().num_bullets},
            {"bullet_type": "wave",             "spawn_time":0, "init_delay": WaveBullet().delay,            "delay": 0, "count": WaveBullet().num_bullets},
            {"bullet_type": "expanding_spiral", "spawn_time":0, "init_delay": ExpandingSpiralBullet().delay, "delay": 0, "count": ExpandingSpiralBullet().num_bullets},
            {"bullet_type": "bouncing_bullet",  "spawn_time":0, "init_delay": BouncingBullet().delay,        "delay": 0, "count": BouncingBullet().num_bullets},
            {"bullet_type": "sin_wave",         "spawn_time":0, "init_delay": 16000,                         "delay": 0, "count": SinWaveBullet().num_bullets},
            ]
        self.sequential_spawns = [
            # {"bullet_type": "spiral",           "spawn_time":0, "init_delay": SpiralBullet().delay,          "delay": 100, "count": SpiralBullet().num_bullets, "spawned": 0},

        ]  # Danh sách các viên đạn spawn tuần tự
    
    
    
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

    def get_random_corner(self) -> tuple[int, int]:
        corners = [(0, 0), (SCREEN_WIDTH, 0), 
                   (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)]
        return random.choice(corners)
    
    def create_bullet_type(self, x, y, spawn_event: dict):

        # if spawn_event["bullet_type"] == "ring":
        #     angle_step = 2 * math.pi / RingBullet().num_bullets
        #     new_bullets = [Bullet(x, y, i * angle_step, RingBullet().speed, RingBullet().radius, color=RingBullet().color) 
        #                for i in range(RingBullet().num_bullets)]
        #     self.bullets.add(*new_bullets)

        # giống ring
        if spawn_event["bullet_type"] == "spiral":
            base_angle = math.radians(self.angle_offset)
            angle_step = 2 * math.pi / SpiralBullet().num_bullets
            new_bullets = [Bullet(x, y, base_angle + i * angle_step, SpiralBullet().speed, SpiralBullet().radius, fade=SpiralBullet().fade, color=SpiralBullet().color) 
                        for i in range(SpiralBullet().num_bullets)]
            self.bullets.add(*new_bullets)
            self.angle_offset += SpiralBullet().rotation_speed

        # oke
        # if spawn_event["bullet_type"] == "targeted_shot":
        #     angle = math.atan2(self.player.x - y, self.player.y - x)
        #     self.bullets.add(Bullet(x, y, angle, DEFAULT_BULLET_SPEED, RingBullet().radius, RingBullet().color))

        # giống ring
        # if spawn_event["bullet_type"] == "rotating_ring":
        #     base_angle = math.radians(self.angle_offset)
        #     angle_step = 2 * math.pi / RotatingRingBullet().num_bullets
        #     new_bullets = [Bullet(x, y, base_angle + i * angle_step, RotatingRingBullet().speed, RotatingRingBullet().radius, fade=RotatingRingBullet().fade, color=RotatingRingBullet().color)
        #                 for i in range(RotatingRingBullet().num_bullets)]
        #     self.bullets.add(*new_bullets)
        #     self.angle_offset += RotatingRingBullet().rotation_speed

        # giống ring
        # if spawn_event["bullet_type"] == "wave":
        #     angle_step = 2 * math.pi / WaveBullet().num_bullets
        #     new_bullets = [Bullet(x, y, i * angle_step, WaveBullet().speed, WaveBullet().radius, color=WaveBullet().color)
        #                 for i in range(RingBullet().num_bullets)]
        #     self.bullets.add(*new_bullets)

        # ???
        # if spawn_event["bullet_type"] == "expanding_spiral":
        #     angle_step = 2 * math.pi / ExpandingSpiralBullet().num_bullets
        #     new_bullets = [Bullet(x, y, i * angle_step, ExpandingSpiralBullet().speed + i * ExpandingSpiralBullet().speed_increment, ExpandingSpiralBullet().radius, color=ExpandingSpiralBullet().color)
        #                 for i in range(RingBullet().num_bullets)]
        #     self.bullets.add(*new_bullets)
            
        # if spawn_event["bullet_type"] == "bouncing_bullet":
        #     angle_step = 2 * math.pi / BouncingBullet().num_bullets
        #     new_bullets = [Bullet(x, y, i * angle_step, BouncingBullet().speed, BouncingBullet().radius, color=BouncingBullet().color, bouncing=True)
        #                 for i in range(RingBullet().num_bullets)]
        #     self.bullets.add(*new_bullets)

        # if spawn_event["bullet_type"] == "sin_wave":
        #     if ((x,y) == (0,0)):
        #         angle = -math.pi/4
        #     elif ((x,y) == (SCREEN_WIDTH, 0)):
        #         angle = math.pi/4   
        #     elif ((x,y) == (0, SCREEN_HEIGHT)):
        #         angle = math.pi * 3 / 4
        #     elif ((x,y) == (SCREEN_WIDTH, SCREEN_HEIGHT)):
        #         angle = math.pi * 5 / 4
        #     new_bullets = [Bullet(x+i, y+i, SinWaveBullet().speed, SinWaveBullet().radius, color=SinWaveBullet().color)
        #                 for i in range(SinWaveBullet().num_bullets)]
        #     self.bullets.add(*new_bullets)
        
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
        bullets = []
        start_radius_square: float = start_radius ** 2
        end_radius_square: float = end_radius ** 2

        for bullet in self.bullets:
            distance_square = (self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2
            if start_radius_square <= distance_square <= end_radius_square:
                bullets.append(bullet)

        return bullets
    
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
    
    def update(self, current_time: int):
        self.bullets.update()
        for event in self.spawn_event:
            x, y = self.get_random_corner()
            if current_time >= event["spawn_time"] + event["init_delay"]:
                event["spawn_time"] = current_time
                if event["delay"] == 0:  
                    # Nếu delay = 0, spawn tất cả đạn cùng lúc
                    self.create_bullet_type(x, y, event)
                else:
                #     # Nếu có delay, thêm vào danh sách spawn tuần tự
                    self.sequential_spawns.append(event)


        for seq_event in self.sequential_spawns[:]:
            if current_time >= seq_event["spawn_time"] + seq_event['spawned'] * seq_event['delay']:
                x, y = 300, 300
                base_angle = 0
                angle_step = 2 * math.pi / seq_event["count"]
                self.bullets.add(Bullet(x, y, seq_event["spawned"] * angle_step, SpiralBullet().speed, SpiralBullet().radius, fade=SpiralBullet().fade, color=SpiralBullet().color))
                seq_event["spawned"] += 1

                if seq_event["spawned"] >= seq_event["count"]:
                    self.sequential_spawns.remove(seq_event)

        for bullet in self.bullets.copy():  # Lọc đạn ra ngoài màn hình
            if bullet.x < 0 or bullet.x > SCREEN_WIDTH or bullet.y < 0 or bullet.y > SCREEN_HEIGHT:
                self.bullets.remove(bullet)


        

    def draw(self, screen):
        for bullet in self.bullets:
            bullet.draw(screen)