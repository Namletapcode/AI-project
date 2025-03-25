import pygame
import math
import random
import numpy as np
from PIL import Image, ImageDraw
from typing import TYPE_CHECKING
from settings import DodgeMethod, DrawSectorMethod, USE_BOT, BOT_ACTION, FILTER_MOVE_INTO_WALL, SCAN_RADIUS, DRAW_SECTOR_METHOD

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

    def draw_sector_use_PIL(self, surface: pygame.Surface, radius: int, from_angle:float, to_angle: float, color: tuple):
        """Vẽ hình quạt (pieslice) bằng PIL rồi chuyển sang pygame."""
        size = (radius * 2, radius * 2)  # Kích thước ảnh
        image = Image.new("RGBA", size, (0, 0, 0, 0))  # Ảnh trong suốt
        draw = ImageDraw.Draw(image)

        # Vẽ hình quạt với PIL
        bbox = (0, 0, size[0], size[1])
        draw.pieslice(bbox, start=math.degrees(from_angle), end=math.degrees(to_angle), fill=color)

        # Chuyển sang pygame
        mode = image.mode
        data = image.tobytes()
        pygame_image = pygame.image.fromstring(data, image.size, mode)

        # Blit hình quạt lên surface
        surface.blit(pygame_image, (self.player.x - radius, self.player.y - radius))
        
    def draw_sector_use_polygon(self, surface: pygame.Surface, radius: int, from_angle:float, to_angle: float, color: tuple, segments: int = 5):
        """
        Vẽ hình quạt mượt bằng cách sử dụng polygon với nhiều điểm trên cung tròn.
        
        - surface: màn hình pygame
        - center: (x, y) của nhân vật
        - radius: bán kính hình quạt
        - start_angle, end_angle: góc bắt đầu và kết thúc (độ)
        - color: màu sắc (RGB)
        - segments: cạnh trên cung tròn (tăng để mượt hơn)
        """
        points = [(self.player.x, self.player.y)]
        for i in range(segments + 1):
            angle = from_angle + (to_angle - from_angle) * (i / segments)
            x = self.player.x + radius * math.cos(angle)
            y = self.player.y - radius * math.sin(angle)
            points.append((x, y))
            # pygame.draw.line(surface, (255, 255, 255), (self.player.x, self.player.y), (x, y))
        pygame.draw.polygon(surface, color, points)  # Vẽ hình quạt
        
    def draw_sector(self, surface, radius, index, color, num_sectors=8, draw_method = DrawSectorMethod.USE_POLYGON):
        sector_angle = 2 * math.pi / num_sectors  # Góc của mỗi nan quạt
        start_angle = -sector_angle / 2
        from_angle = start_angle + index * sector_angle
        to_angle = from_angle + sector_angle
        if draw_method == DrawSectorMethod.USE_POLYGON:
            self.draw_sector_use_polygon(surface, radius, from_angle, to_angle, color)
            return
        if draw_method == DrawSectorMethod.USE_TRIANGLE:
            # Tính tọa độ hai điểm ngoài cung tròn
            x1 = self.player.x + radius * math.cos(from_angle)
            y1 = self.player.y - radius * math.sin(from_angle)
            x2 = self.player.x + radius * math.cos(to_angle)
            y2 = self.player.y - radius * math.sin(to_angle)

            # Vẽ hình quạt bằng tam giác nối với tâm
            points = [(self.player.x, self.player.y), (x1, y1), (x2, y2)]
            pygame.draw.polygon(self.screen, color, points)
            return
        if draw_method == DrawSectorMethod.USE_TRIANGLE_AND_ARC:
            # Tính tọa độ hai điểm ngoài cung tròn
            x1 = self.player.x + radius * math.cos(from_angle)
            y1 = self.player.y - radius * math.sin(from_angle)
            x2 = self.player.x + radius * math.cos(to_angle)
            y2 = self.player.y - radius * math.sin(to_angle)

            # Vẽ hình quạt bằng tam giác nối với tâm
            points = [(self.player.x, self.player.y), (x1, y1), (x2, y2)]
            pygame.draw.polygon(self.screen, color, points)
            
            # Vùng chứa vòng tròn
            arc_rect = pygame.Rect(self.player.x - radius, self.player.y - radius, 2 * radius, 2 * radius)
            
            pygame.draw.arc(self.screen, color, arc_rect, from_angle, to_angle, 5)                
            return
        if draw_method == DrawSectorMethod.USE_PIL:
            self.draw_sector_use_PIL(self.screen, radius, from_angle, to_angle, color)
            return

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
    
    def draw_sectors(self, radius, num_sectors=None, draw_method = DrawSectorMethod.USE_POLYGON):
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
                    self.draw_sector(self.screen, radius, i, color, num_sectors, draw_method)
                    
    def draw_vison(self):
        self.player.draw_surround_circle(SCAN_RADIUS)
        self.draw_sectors(SCAN_RADIUS, None, DRAW_SECTOR_METHOD)
        self.game.bullet_manager.color_in_radius(SCAN_RADIUS, (128, 0, 128))
        best_direction_index = np.argmax(self.action)
        if best_direction_index != 8:
            self.draw_sector(self.screen, 50, best_direction_index, (0, 255, 0))
    
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
        return USE_BOT or BOT_ACTION