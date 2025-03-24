import pygame
import math
import random
import numpy as np
from PIL import Image, ImageDraw
from typing import TYPE_CHECKING
from settings import DodgeMethod, DrawSectorMethod, USE_BOT

if TYPE_CHECKING:
    from game import Game

class GameBot:
    def __init__(self, game: "Game", method = DodgeMethod.FURTHEST_SAFE_DIRECTION):
        self.method = method
        self.game = game
        self.player = self.game.player
        self.bullets = self.game.bullet_manager.bullets
        self.screen = self.game.screen
        self.action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  if USE_BOT else None

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

    def classify_bullets_into_sectors(self, bullets, num_sectors=8, start_angle=-math.pi/8) -> np.ndarray:
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
    
    def draw_sectors(self, radius, num_sectors=8, draw_method = DrawSectorMethod.USE_POLYGON):
        # Lấy danh sách đạn trong bán kính d
        bullets_in_radius = self.game.bullet_manager.bullets_in_radius(self.screen, radius)
        
        # Phân loại đạn vào các nan quạt
        sector_flags = self.classify_bullets_into_sectors(bullets_in_radius)

        for i in range(num_sectors):
            # Chọn màu: Vàng nếu có đạn, Trắng nếu không
            color = (255, 255, 0) if sector_flags[i] else (255, 255, 255)
            
            # Vẽ viền cung tròn
            if sector_flags[i]:
                self.draw_sector(self.screen, radius, i, color, num_sectors, draw_method)
    
    def get_safe_action(self, radius: int = 70):
        bullets_near_player = self.game.bullet_manager.bullets_in_radius(self.screen, radius)
        self.reset_action()
        if len(bullets_near_player) == 0:
            return 8
        if self.method == DodgeMethod.FURTHEST_SAFE_DIRECTION:
            # Đánh giá an toàn cho mỗi hướng
            safe_scores = []
            for direction in self.player.directions:
                new_pos = self.player.direction_to_position(direction)  # Vị trí nếu di chuyển theo hướng này
                safe_score = sum(
                    (new_pos.x - bullet.x) ** 2 + (new_pos.y - bullet.y) ** 2  # Càng xa càng an toàn
                    for bullet in bullets_near_player
                )
                safe_scores.append(safe_score)
            # Chọn hướng có điểm nguy hiểm thấp nhất
            best_direction_index = safe_scores.index(max(safe_scores))
            self.draw_sector(self.screen, 50, best_direction_index, (0, 255, 0))
        elif self.method == DodgeMethod.LEAST_DANGER_PATH:
            # Đánh giá an toàn cho mỗi hướng
            danger_scores = []
            for direction in self.player.directions:
                new_pos = self.player.direction_to_position(direction)  # Vị trí nếu di chuyển theo hướng này
                danger_score = sum(
                    1 / ((new_pos.x - bullet.x) ** 2 + (new_pos.y - bullet.y) ** 2 + 1) # Càng gần viên đạn, nguy cơ càng cao
                    for bullet in bullets_near_player
                )
                danger_scores.append(danger_score)
            # Chọn hướng có điểm nguy hiểm thấp nhất
            best_direction_index = danger_scores.index(min(danger_scores))
            self.draw_sector(self.screen, 50, best_direction_index, (0, 255, 0))
        elif self.method == DodgeMethod.RANDOM_SAFE_ZONE:
            sector_flags = self.classify_bullets_into_sectors(bullets_near_player)
            list_move = []
            for i in range(len(sector_flags)):
                if not sector_flags[i]:
                    list_move.append(i)
            if list_move:
                best_direction_index = random.choice(list_move)
                self.draw_sector(self.screen, 50, best_direction_index, (0, 255, 0))
                
        if self.action is not None:
            # Cập nhật self.action theo dạng One-Hot
            self.action[best_direction_index] = 1  
        return best_direction_index

    def reset_action(self):
        if USE_BOT:
            self.action[:] = 0
            self.action[-1] = 1 # phần tử cuối ứng với đứng yên gán mặc định bằng 1
        else: self.actio = None