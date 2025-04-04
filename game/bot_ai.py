import pygame
import math
import random
import numpy as np
from typing import TYPE_CHECKING
from settings import (DodgeMethod, BOT_ACTION, FILTER_MOVE_INTO_WALL, SCAN_RADIUS,
                      USE_WALL_PENALTY, WALL_PENALTY_BIAS, WALL_MARGIN, BOX_LEFT, BOX_SIZE, BOX_TOP)
from help_methods import draw_sector

if TYPE_CHECKING:
    from game import Game

class GameBot:
    def __init__(self, game: "Game", method = DodgeMethod.FURTHEST_SAFE_DIRECTION):
        self.method = method
        self.game = game
        self.player = self.game.player
        self.bullets = self.game.bullet_manager.bullets
        self.screen = self.game.screen
        self.action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  # if self.is_activate() else None
    def classify_bullets_into_sectors(self, bullets, num_sectors=8, start_angle=-math.pi/8) -> np.ndarray: # temporally not in use
        sector_flags = np.zeros(num_sectors)
        sector_angle = 2 * math.pi / num_sectors  # Góc mỗi nan quạt

        for bullet in bullets:
            # Tính góc của viên đạn so với nhân vật
            angle = math.atan2(self.player.y - bullet.y, bullet.x - self.player.x)

            # Chỉnh lại góc về phạm vi [0, 360)
            angle = (angle - start_angle) % (2 * math.pi)

            # Xác định nan quạt nào chứa viên đạn
            sector_index = int(angle // sector_angle)
            sector_flags[sector_index] = 1

        return sector_flags
    
    def draw_sectors(self, radius, num_sectors=None):
        if num_sectors is None:
            # Lấy danh sách đạn trong bán kính d
            bullets_in_radius = self.game.bullet_manager.get_bullet_in_range(radius)
            
            # Phân loại đạn vào các nan quạt
            sector_flags = self.game.bullet_manager.get_converted_regions(bullets_in_radius)

            num_sectors = len(sector_flags)

            for i in range(num_sectors):
                # Chọn màu: Vàng nếu có đạn, Trắng nếu không
                color = (255, 255, 0) if sector_flags[i] else (255, 255, 255)
                
                # Vẽ viền cung tròn
                if sector_flags[i]:
                    draw_sector(self.screen, self.player.x, self.player.y, radius, i, color, num_sectors)
                    
    def draw_vison(self):
        self.player.draw_surround_circle(SCAN_RADIUS)
        self.draw_sectors(SCAN_RADIUS, None)
        self.game.bullet_manager.color_in_radius(SCAN_RADIUS, (128, 0, 128))
        best_direction_index = np.argmax(self.action)
        if best_direction_index != 8:
            draw_sector(self.screen, self.player.x, self.player.y, 50, best_direction_index, (0, 255, 0))
    
    def update(self):
        radius = SCAN_RADIUS
        bullets_near_player = self.game.bullet_manager.get_bullet_in_range(radius)
        self.reset_action()

        if len(bullets_near_player) == 0:
            return 8
        if self.method == DodgeMethod.FURTHEST_SAFE_DIRECTION:
            best_direction_index = self.furthest_safe(bullets_near_player)

        elif self.method == DodgeMethod.LEAST_DANGER_PATH:
            best_direction_index = self.least_danger(bullets_near_player)
            
        elif self.method == DodgeMethod.LEAST_DANGER_PATH_ADVANCED:
            best_direction_index = self.least_danger_advanced()

        elif self.method == DodgeMethod.RANDOM_SAFE_ZONE:
            best_direction_index = self.random_move(bullets_near_player)

        elif self.method == DodgeMethod.OPPOSITE_THREAT_DIRECTION:
            best_direction_index = self.opposite_threat(bullets_near_player)
            
        
        # Cập nhật self.action theo dạng One-Hot
        self.action[best_direction_index] = 1
        self.action[8] = 0
        
        return best_direction_index
    
    def furthest_safe(self, bullets_near_player):
        # Đánh giá an toàn cho mỗi hướng
        safe_scores = []
        for direction in self.player.directions:
            new_pos = self.player.direction_to_position(direction)  # Vị trí nếu di chuyển theo hướng này
            safe_score = sum(
                (new_pos.x - bullet.x) ** 2 + (new_pos.y - bullet.y) ** 2  # Càng xa càng an toàn
                for bullet in bullets_near_player
            )
            safe_scores.append(safe_score)
            
        if USE_WALL_PENALTY:
            # Áp dụng soft penalty để bot không lại gần tường
            danger_scores = self.apply_soft_wall_penalty([0.0] * 9)
            for i in range(8): safe_scores[i] -= danger_scores[i]
            
        # nếu có lọc hướng di chuyển đâm vào hộp
        if FILTER_MOVE_INTO_WALL:
            near_wall_info = self.player.get_near_wall_info()
            if near_wall_info[0]:
                safe_scores[1] = safe_scores[2] = safe_scores[3] = 0
            if near_wall_info[1]:
                safe_scores[7] = safe_scores[0] = safe_scores[1] = 0
            if near_wall_info[2]:
                safe_scores[5] = safe_scores[6] = safe_scores[7] = 0
            if near_wall_info[3]:
                safe_scores[3] = safe_scores[4] = safe_scores[5] = 0
        # Chọn hướng có điểm nguy hiểm thấp nhất
        best_direction_index = safe_scores.index(max(safe_scores))

        return best_direction_index

    def least_danger(self, bullets_near_player):
        # Đánh giá an toàn cho mỗi hướng
        danger_scores = []
        for direction in self.player.directions:
            new_pos = self.player.direction_to_position(direction)  # Vị trí nếu di chuyển theo hướng này
            danger_score = sum(
                1 / ((new_pos.x - bullet.x) ** 2 + (new_pos.y - bullet.y) ** 2 + 1) # Càng gần viên đạn, nguy cơ càng cao
                for bullet in bullets_near_player
            )
            danger_scores.append(danger_score)
        
        if USE_WALL_PENALTY:
            # Áp dụng soft penalty để bot không lại gần tường
            danger_scores = self.apply_soft_wall_penalty(danger_scores)
        
        # nếu có lọc hướng di chuyển đâm vào hộp
        if FILTER_MOVE_INTO_WALL:
            near_wall_info = self.player.get_near_wall_info()
            if near_wall_info[0]:
                danger_scores[1] = danger_scores[2] = danger_scores[3] = float('inf')
            if near_wall_info[1]:
                danger_scores[7] = danger_scores[0] = danger_scores[1] = float('inf')
            if near_wall_info[2]:
                danger_scores[5] = danger_scores[6] = danger_scores[7] = float('inf')
            if near_wall_info[3]:
                danger_scores[3] = danger_scores[4] = danger_scores[5] = float('inf')
        # Chọn hướng có điểm nguy hiểm thấp nhất
        best_direction_index = danger_scores.index(min(danger_scores))

        return best_direction_index
    
    def least_danger_advanced(self):
        danger_scores = [0.0] * 9  # 8 hướng + đứng yên

        # Các zone + future_ticks tương ứng
        zones = [
            (0, SCAN_RADIUS/2, 0),     # Rất gần → không dự đoán, xử lý gấp
            (SCAN_RADIUS/2, SCAN_RADIUS, 5),    # Gần → dự đoán ngắn
        ]

        for (start_r, end_r, ticks) in zones:
            bullets_zone = self.game.bullet_manager.get_bullet_in_range(end_r, start_r)
            partial_scores = self.predict_future_danger(bullets_zone, future_ticks=ticks)
            # Gộp điểm nguy hiểm lại theo hướng
            for i in range(9):
                danger_scores[i] += partial_scores[i]
                
        if USE_WALL_PENALTY:
            # Áp dụng soft penalty để bot không lại gần tường
            danger_scores = self.apply_soft_wall_penalty(danger_scores)

        # Lọc hướng đâm vào tường nếu bật flag
        if FILTER_MOVE_INTO_WALL:
            wall_info = self.player.get_near_wall_info()
            if wall_info[0]:
                danger_scores[1] = danger_scores[2] = danger_scores[3] = float('inf')
            if wall_info[1]:
                danger_scores[7] = danger_scores[0] = danger_scores[1] = float('inf')
            if wall_info[2]:
                danger_scores[5] = danger_scores[6] = danger_scores[7] = float('inf')
            if wall_info[3]:
                danger_scores[3] = danger_scores[4] = danger_scores[5] = float('inf')

        return danger_scores.index(min(danger_scores))

    def predict_future_danger(self, bullets_near_player, future_ticks=10) -> list[float]:
        # Đánh giá mức độ nguy hiểm cho mỗi hướng
        danger_scores = []

        for direction in self.player.directions:
            new_pos = self.player.direction_to_position(direction)  # Vị trí nếu di chuyển theo hướng này
            danger_score = sum(
                1 / ((new_pos.x - (bullet.x + future_ticks * (math.cos(bullet.angle) * bullet.speed))) ** 2 +
                     (new_pos.y - (bullet.y + future_ticks * (math.sin(bullet.angle) * bullet.speed))) ** 2 + 1) # Càng gần viên đạn, nguy cơ càng cao
                for bullet in bullets_near_player
            )
            danger_scores.append(danger_score)
        return danger_scores
    
    # Phạt mềm nếu di chuyển lại gần tường
    def apply_soft_wall_penalty(self, danger_scores, margin=WALL_MARGIN):
        wall_bias_strength = WALL_PENALTY_BIAS  # Số càng cao, càng "ghét" tường

        for i, direction in enumerate(self.player.directions):
            new_pos = self.player.direction_to_position(direction)
            penalty = 0

            if new_pos.x < BOX_LEFT + margin:
                d = BOX_LEFT + margin - new_pos.x
                penalty += wall_bias_strength * (d / margin)
            elif new_pos.x > BOX_LEFT + BOX_SIZE - margin:
                d = new_pos.x - (BOX_LEFT + BOX_SIZE - margin)
                penalty += wall_bias_strength * (d / margin)

            if new_pos.y < BOX_TOP + margin:
                d = BOX_TOP + margin - new_pos.y
                penalty += wall_bias_strength * (d / margin)
            elif new_pos.y > BOX_TOP + BOX_SIZE - margin:
                d = new_pos.y - (BOX_TOP + BOX_SIZE - margin)
                penalty += wall_bias_strength * (d / margin)

            danger_scores[i] += penalty

        return danger_scores

    def opposite_threat(self, bullets_near_player):
        sector_flags = self.classify_bullets_into_sectors(bullets_near_player)

        # Tính tổng nguy hiểm của từng nhóm hướng (trái/phải, trên/dưới)
        vertical_threat = sector_flags[5] + sector_flags[6] + sector_flags[7] - (sector_flags[1] + sector_flags[2] + sector_flags[3])
        horizontal_threat = sector_flags[7] + sector_flags[0] + sector_flags[1] - (sector_flags[3] + sector_flags[4] + sector_flags[5])

        # Xác định hướng di chuyển an toàn hơn
        move_y = -1 if vertical_threat > 0 else (1 if vertical_threat < 0 else 0)
        move_x = -1 if horizontal_threat > 0 else (1 if horizontal_threat < 0 else 0)
            
        best_direction_index = self.game.player.directions.index(pygame.Vector2(move_x, move_y))

        return best_direction_index

    def random_move(self, bullets_near_player):
        sector_flags = self.classify_bullets_into_sectors(bullets_near_player)
        list_move = []
        for i in range(len(sector_flags)):
            if not sector_flags[i]:
                list_move.append(i)
        if list_move:
            best_direction_index = random.choice(list_move)

        return best_direction_index

    def reset_action(self):
        self.action[:] = 0
        self.action[-1] = 1 # phần tử cuối ứng với đứng yên gán mặc định bằng 1

    def is_activate(self) -> bool:
        return BOT_ACTION