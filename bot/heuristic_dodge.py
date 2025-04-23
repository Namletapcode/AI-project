import math, random
import numpy as np
import pygame
from configs.bot_config import (
    DodgeAlgorithm, FILTER_MOVE_INTO_WALL, SCAN_RADIUS, USE_COMPLEX_SCANNING,
    USE_WALL_PENALTY, WALL_PENALTY_BIAS, WALL_MARGIN)
from configs.game_config import BOX_LEFT, BOX_SIZE, BOX_TOP
from utils.draw_utils import draw_sector, draw_complex_sector
from bot.base_bot import BaseBot
from game.game_core import Game

class HeuristicDodgeBot(BaseBot):
    is_heuristic = True
    def __init__(self, game: "Game", method = DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED):
        super().__init__(game)
        self.method = method
        self.game = game
        self.surface = game.surface
        # Hành động: 8 hướng + đứng yên (index 8)
        self.action = np.array([0,0,0,0,0,0,0,0,1])
        
    def get_action(self, state: list) -> np.ndarray:
        # state: list bullet in SCAN_RADIUS
        self.reset_action()

        if len(state) == 0:
            self.reset_action()
            return self.action

        method_map = {
            DodgeAlgorithm.FURTHEST_SAFE_DIRECTION: self.furthest_safe,
            DodgeAlgorithm.LEAST_DANGER_PATH: self.least_danger,
            DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED: self.least_danger_advanced,
            DodgeAlgorithm.RANDOM_SAFE_ZONE: self.random_move,
            DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION: self.opposite_threat
        }
        
        # Lấy hàm xử lý né đạn dựa trên self.method từ method_map.
        # Nếu self.method không tồn tại trong map, sử dụng hàm mặc định lambda *_: 8.
        # lambda *_: 8 là một hàm nhận bất kỳ số lượng tham số nào nhưng luôn trả về 8 (vô hiệu hóa hành động).
        func = method_map.get(self.method, lambda *_: 8)
        
        best_direction_index = func(state)

        self.action[best_direction_index] = 1
        self.action[8] = 0
        return self.action

    def furthest_safe(self, bullets_near_player):
        safe_scores = []
        for direction in self.player.directions:
            new_pos = self.player.direction_to_position(direction)
            safe_score = sum(
                (new_pos.x - bullet.x) ** 2 + (new_pos.y - bullet.y) ** 2
                for bullet in bullets_near_player
            )
            safe_scores.append(safe_score)
            
        if USE_WALL_PENALTY:
            danger_scores = self.apply_soft_wall_penalty([0.0] * 9)
            for i in range(8): safe_scores[i] -= danger_scores[i]
        if FILTER_MOVE_INTO_WALL:
            self.filter_wall_directions(safe_scores)
            
        best_direction_index = safe_scores.index(max(safe_scores))
        return best_direction_index

    def least_danger(self, bullets_near_player):
        danger_scores = []
        for direction in self.player.directions:
            new_pos = self.player.direction_to_position(direction)
            danger_score = sum(
                1 / ((new_pos.x - bullet.x) ** 2 + (new_pos.y - bullet.y) ** 2 + 1)
                for bullet in bullets_near_player
            )
            danger_scores.append(danger_score)
            
        if USE_WALL_PENALTY:
            danger_scores = self.apply_soft_wall_penalty(danger_scores)
            
        if FILTER_MOVE_INTO_WALL:
            self.filter_wall_directions(danger_scores, use_inf=True)
            
        best_direction_index = danger_scores.index(min(danger_scores))
        
        return best_direction_index

    def least_danger_advanced(self, _):
        danger_scores = [0.0] * 9
        zones = [
            (0, SCAN_RADIUS/2, 0),
            (SCAN_RADIUS/2, SCAN_RADIUS, 5),
        ]
        for (start_r, end_r, ticks) in zones:
            bullets_zone = self.game.bullet_manager.get_bullet_in_range(end_r, start_r)
            partial_scores = self.predict_future_danger(bullets_zone, future_ticks=ticks)
            for i in range(9):
                danger_scores[i] += partial_scores[i]
                
        if USE_WALL_PENALTY:
            danger_scores = self.apply_soft_wall_penalty(danger_scores)
            
        if FILTER_MOVE_INTO_WALL:
            self.filter_wall_directions(danger_scores, use_inf=True)
            
        return danger_scores.index(min(danger_scores))

    def predict_future_danger(self, bullets_near_player, future_ticks=10) -> list[float]:
        danger_scores = []
        for direction in self.player.directions:
            new_pos = self.player.direction_to_position(direction)
            danger_score = sum(
                1 / ((new_pos.x - (bullet.x + future_ticks * math.cos(bullet.angle) * bullet.speed)) ** 2 +
                     (new_pos.y - (bullet.y + future_ticks * math.sin(bullet.angle) * bullet.speed)) ** 2 + 1)
                for bullet in bullets_near_player
            )
            danger_scores.append(danger_score)
        return danger_scores
    
    def opposite_threat(self, bullets_near_player):
        sector_flags = self.classify_bullets_into_sectors(bullets_near_player)
        vertical_threat = sector_flags[5] + sector_flags[6] + sector_flags[7] - (sector_flags[1] + sector_flags[2] + sector_flags[3])
        horizontal_threat = sector_flags[7] + sector_flags[0] + sector_flags[1] - (sector_flags[3] + sector_flags[4] + sector_flags[5])
        move_y = -1 if vertical_threat > 0 else (1 if vertical_threat < 0 else 0)
        move_x = -1 if horizontal_threat > 0 else (1 if horizontal_threat < 0 else 0)
        best_direction_index = self.game.player.directions.index(pygame.Vector2(move_x, move_y))
        return best_direction_index

    def random_move(self, bullets_near_player):
        sector_flags = self.classify_bullets_into_sectors(bullets_near_player)
        safe_dirs = [i for i, flag in enumerate(sector_flags) if not flag]
        return random.choice(safe_dirs) if safe_dirs else 8

    def apply_soft_wall_penalty(self, danger_scores, margin=WALL_MARGIN):
        for i, direction in enumerate(self.player.directions):
            pos = self.player.direction_to_position(direction)
            penalty = 0
            if pos.x < BOX_LEFT + margin:
                d = BOX_LEFT + margin - pos.x
                penalty += WALL_PENALTY_BIAS * (d / margin)
            elif pos.x > BOX_LEFT + BOX_SIZE - margin:
                d = pos.x - (BOX_LEFT + BOX_SIZE - margin)
                penalty += WALL_PENALTY_BIAS * (d / margin)
            if pos.y < BOX_TOP + margin:
                d = BOX_TOP + margin - pos.y
                penalty += WALL_PENALTY_BIAS * (d / margin)
            elif pos.y > BOX_TOP + BOX_SIZE - margin:
                d = pos.y - (BOX_TOP + BOX_SIZE - margin)
                penalty += WALL_PENALTY_BIAS * (d / margin)
            danger_scores[i] += penalty
        return danger_scores
    
    def filter_wall_directions(self, scores, use_inf=False):
        wall_info = self.player.get_near_wall_info()
        val = float('inf') if use_inf else 0
        if wall_info[0]: scores[1] = scores[2] = scores[3] = val
        if wall_info[1]: scores[7] = scores[0] = scores[1] = val
        if wall_info[2]: scores[5] = scores[6] = scores[7] = val
        if wall_info[3]: scores[3] = scores[4] = scores[5] = val

    def classify_bullets_into_sectors(self, bullets, num_sectors=8, start_angle=-math.pi/8) -> np.ndarray:
        sector_flags = np.zeros(num_sectors)
        sector_angle = 2 * math.pi / num_sectors

        for bullet in bullets:
            # Tính góc của viên đạn so với nhân vật
            angle = math.atan2(self.player.y - bullet.y, bullet.x - self.player.x)
            # Chỉnh lại góc về phạm vi [0, 360)
            angle = (angle - start_angle) % (2 * math.pi)
            # Xác định nan quạt nào chứa viên đạn
            sector_index = int(angle // sector_angle)
            sector_flags[sector_index] = 1

        return sector_flags
    
    def draw_vision(self):
        self.player.draw_surround_circle(SCAN_RADIUS)
        if USE_COMPLEX_SCANNING:
            self.draw_complex_sectors(SCAN_RADIUS)
        else:
            self.draw_simple_sectors(SCAN_RADIUS)
        self.game.bullet_manager.color_in_radius(SCAN_RADIUS, (128,0,128))
        best_direction_index = np.argmax(self.action)
        if best_direction_index != 8:
            draw_sector(self.surface, self.player.x, self.player.y, 50, best_direction_index, (0, 255, 0))
    
    def draw_simple_sectors(self, radius: int):
        """Vẽ các sector đơn giản (chỉ chia theo góc)"""
        bullets_in_radius = self.game.bullet_manager.get_bullet_in_range(radius)
        sector_flags = self.game.bullet_manager.get_simple_regions(bullets_in_radius)
        num_sectors = len(sector_flags)
        for i in range(num_sectors):
            # Chọn màu: Vàng nếu có đạn, Trắng nếu không
            color = (255,255,0) if sector_flags[i] else (255,255,255)
            if sector_flags[i]:
                draw_sector(
                    self.surface, self.player.x, self.player.y,
                    radius, i, color, num_sectors)
    
    def draw_complex_sectors(self, radius: int, num_angle_divisions: int = 8, num_radius_divisions: int = 3):
        """Vẽ các sector phức tạp (chia theo cả góc và bán kính)"""
        bullets_in_radius = self.game.bullet_manager.get_bullet_in_range(radius)
        sector_flags = self.game.bullet_manager.get_complex_regions(bullets_in_radius, num_angle_divisions, num_radius_divisions)
        for i in range(len(sector_flags)):
            color = (255,255,0) if sector_flags[i] else (255,255,255)
            if sector_flags[i]:
                draw_complex_sector(
                    self.surface, self.player.x, self.player.y, i,
                    radius, num_angle_divisions, num_radius_divisions, color)

    def reset_action(self):
        self.action[:] = 0
        self.action[-1] = 1