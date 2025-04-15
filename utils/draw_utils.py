from math import sin, cos, asin, sqrt, degrees, pi
from PIL import Image, ImageDraw
import pygame
from configs.game_config import DrawSectorMethod, DRAW_SECTOR_METHOD

# ...existing code...

def draw_complex_sector(
    surface: pygame.Surface, point_x: int, point_y: int, index: int,
    radius: float, num_angle_divisions: int = 16, num_radius_divisions: int = 3,
    color: tuple = (255, 255, 0), segments: int = 3):
    '''
    Vẽ một phần tử trong hình tròn được chia thành num_angle_divisions*num_radius_divisions phần.
    
    Args:
        point_x, point_y: Tọa độ tâm
        index: Chỉ số phần tử (0 đến num_angle_divisions*num_radius_divisions-1)
        radius: Bán kính hình tròn
        num_angle_divisions: Số phần chia theo góc
        num_radius_divisions: Số phần chia theo bán kính
        color: Màu sắc
        segments: cạnh trên cung tròn (tăng để mượt hơn)
    '''
    # Xác định phần tử thuộc vòng nào và góc nào
    ring_index = index // num_angle_divisions  # Chỉ số vòng (0 đến num_radius_divisions-1)
    angle_index = index % num_angle_divisions   # Chỉ số góc trong vòng (0 đến num_angle_divisions-1)
    
    # Tính bán kính trong và ngoài của phần tử
    inner_radius = (ring_index * radius) / num_radius_divisions
    outer_radius = ((ring_index + 1) * radius) / num_radius_divisions
    
    # Tính góc bắt đầu và kết thúc
    sector_angle = 2 * pi / num_angle_divisions
    from_angle = -sector_angle / 2 + angle_index * sector_angle
    to_angle = from_angle + sector_angle
    
    # Tạo các điểm cho polygon
    points = []
    
    # Thêm các điểm ở vòng trong
    if inner_radius > 0:
        for i in range(segments + 1):
            angle = from_angle + (to_angle - from_angle) * (i / segments)
            x_inner = point_x + inner_radius * cos(angle)
            y_inner = point_y - inner_radius * sin(angle)
            points.append((x_inner, y_inner))
    else: points.append((point_x, point_y))  # Nếu không có vòng trong, thêm tâm
    
    # Thêm các điểm ở vòng ngoài
    for i in range(segments, -1, -1):
        angle = from_angle + (to_angle - from_angle) * (i / segments)
        x_outer = point_x + outer_radius * cos(angle)
        y_outer = point_y - outer_radius * sin(angle)
        points.append((x_outer, y_outer))
    
    # Vẽ polygon
    pygame.draw.polygon(surface, color, points)

    # Vẽ viền (tùy chọn)
    # pygame.draw.lines(surface, (0, 0, 0), True, points, 1)
    
def draw_sector(surface: pygame.Surface, point_x: int, point_y: int, radius: int, index: int, color, num_sectors=8, draw_method = DRAW_SECTOR_METHOD):
    sector_angle = 2 * pi / num_sectors  # Góc của mỗi nan quạt
    start_angle = -sector_angle / 2
    from_angle = start_angle + index * sector_angle
    to_angle = from_angle + sector_angle
    if draw_method == DrawSectorMethod.USE_POLYGON:
        draw_sector_use_polygon(surface, point_x, point_y, radius, from_angle, to_angle, color)
        return
    if draw_method == DrawSectorMethod.USE_TRIANGLE:
        # Tính tọa độ hai điểm ngoài cung tròn
        x1 = point_x + radius * cos(from_angle)
        y1 = point_y - radius * sin(from_angle)
        x2 = point_x + radius * cos(to_angle)
        y2 = point_y - radius * sin(to_angle)

        # Vẽ hình quạt bằng tam giác nối với tâm
        points = [(point_x, point_y), (x1, y1), (x2, y2)]
        pygame.draw.polygon(surface, color, points)
        return
    if draw_method == DrawSectorMethod.USE_TRIANGLE_AND_ARC:
        # Tính tọa độ hai điểm ngoài cung tròn
        x1 = point_x + radius * cos(from_angle)
        y1 = point_y - radius * sin(from_angle)
        x2 = point_x + radius * cos(to_angle)
        y2 = point_y - radius * sin(to_angle)

        # Vẽ hình quạt bằng tam giác nối với tâm
        points = [(point_x, point_y), (x1, y1), (x2, y2)]
        pygame.draw.polygon(surface, color, points)
        
        # Vùng chứa vòng tròn
        arc_rect = pygame.Rect(point_x - radius, point_y - radius, 2 * radius, 2 * radius)
        
        pygame.draw.arc(surface, color, arc_rect, from_angle, to_angle, 10)
        return
    if draw_method == DrawSectorMethod.USE_PIL:
        draw_sector_use_PIL(surface, point_x, point_y, radius, from_angle, to_angle, color)
        return
    
def draw_sector_use_PIL(surface: pygame.Surface, point_x: int, point_y: int, radius: int, from_angle:float, to_angle: float, color: tuple):
    """Vẽ hình quạt (pieslice) bằng PIL rồi chuyển sang pygame."""
    size = (radius * 2, radius * 2)  # Kích thước ảnh
    image = Image.new("RGBA", size, (0, 0, 0, 0))  # Ảnh trong suốt
    draw = ImageDraw.Draw(image)

    # Vẽ hình quạt với PIL
    bbox = (0, 0, size[0], size[1])
    draw.pieslice(bbox, start=degrees(from_angle), end=degrees(to_angle), fill=color)

    # Chuyển sang pygame
    mode = image.mode
    data = image.tobytes()
    pygame_image = pygame.image.fromstring(data, image.size, mode)

    # Blit hình quạt lên surface
    surface.blit(pygame_image, (point_x - radius, point_y - radius))

def draw_sector_use_polygon(surface: pygame.Surface, point_x: int, point_y: int, radius: int, from_angle:float, to_angle: float, color: tuple, segments: int = 5):
    """
    Vẽ hình quạt mượt bằng cách sử dụng polygon với nhiều điểm trên cung tròn.
    
    - surface: màn hình pygame
    - center: (x, y) của nhân vật
    - radius: bán kính hình quạt
    - start_angle, end_angle: góc bắt đầu và kết thúc (độ)
    - color: màu sắc (RGB)
    - segments: cạnh trên cung tròn (tăng để mượt hơn)
    """
    points = [(point_x, point_y)]
    for i in range(segments + 1):
        angle = from_angle + (to_angle - from_angle) * (i / segments)
        x = point_x + radius * cos(angle)
        y = point_y - radius * sin(angle)
        points.append((x, y))
    pygame.draw.polygon(surface, color, points)  # Vẽ hình quạt

def rotate_point(px, py, cx, cy, angle):
        """Xoay điểm (px, py) quanh tâm (cx, cy) một góc angle (radian)."""
        cos_theta = cos(angle)
        sin_theta = sin(angle)
        dx, dy = px - cx, py - cy  # Vector từ tâm đến điểm
        new_x = cx + dx * cos_theta - dy * sin_theta
        new_y = cy + dx * sin_theta + dy * cos_theta
        return new_x, new_y

def draw_water_drop(surface: pygame.Surface, object: object):
    if len(object.trail) < 2:
        return  # Không đủ điểm để vẽ

    P_tail = object.trail[0]  # Đít của giọt nước

    # Tính khoảng cách d
    dx, dy = object.x - P_tail[0], object.y - P_tail[1]
    d = sqrt(dx**2 + dy**2)
    mid_point_x = (object.x + P_tail[0])/2
    mid_point_y = (object.y + P_tail[1])/2

    if d <= object.radius:
        return  # Tránh lỗi chia 0 khi d quá nhỏ

    theta = 2 * asin(object.radius / d)

    # Quay P_head quanh P_tail góc ±theta/2
    T1 = rotate_point(object.x, object.y, mid_point_x, mid_point_y, theta)
    T2 = rotate_point(object.x, object.y, mid_point_x, mid_point_y, -theta)
    
    trail_color = tuple(c * 0.5 for c in object.color)

    pygame.draw.polygon(surface, trail_color, [P_tail, T1, T2])