from enum import Enum
from collections import namedtuple

SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
GAME_SPEED = 1.0                    # Tốc độ game (1.0 là tốc độ bình thường)
FPS = 60
BASE_UPS = 60                       # Tốc độ cập nhật game (updates per second) ở GAME_SPEED = 1.0 (Phải là bội số của FPS)
UPDATE_DELTA_TIME = 1 / BASE_UPS    # delta time cho các lần cập nhật game (seconds)
UPS =  BASE_UPS * GAME_SPEED        # Tốc độ cập nhật game (updates per second)
dt_max = 3 / FPS

PLAYER_SPEED = 200                  # Tốc độ di chuyển của player (pixel/s)
DEFAULT_BULLET_SPEED = 150          # Tốc độ đạn (pixel/s)

DEFAULT_BULLET_RADIUS = 5
BOX_SIZE = 500
BOX_TOP = (SCREEN_HEIGHT - BOX_SIZE) / 2
BOX_LEFT = (SCREEN_WIDTH - BOX_SIZE) / 2

class DrawSectorMethod(Enum):
    USE_POLYGON = 0
    USE_TRIANGLE = 1
    USE_TRIANGLE_AND_ARC = 2
    USE_PIL = 3

DRAW_SECTOR_METHOD = DrawSectorMethod.USE_POLYGON

# Cấu hình vệt mờ
DISPLAY_PLAYER_TRAIL = True
DISPLAY_BULLET_TRAIL = False
TRAIL_MAX_LENGTH = 10  # Giới hạn số điểm lưu trong vệt mờ

USE_BULLET_COLORS = False
# Colors (R, G, B)
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
RED     = (255, 0, 0)
GREEN   = (0, 255, 0)
BLUE    = (0, 0, 255)
YELLOW  = (255, 255, 0)
PURPLE  = (128, 0, 128)
CYAN    = (0, 255, 255)
MAGENTA = (255, 0, 255)

BulletConfig = namedtuple('BulletConfig', [
    'num_bullets',
    'speed', # Tốc độ đạn (pixel/s) ở GAME_SPEED = 1.0
    'delay',
    'radius',
    'color',
    'fade',
    'rotation_speed',  # cho rotating và spiral
    'wave_amplitude',  # cho wave
    'speed_increment'  # cho expanding
], defaults=[0] * 9)  # Set default 0 cho tất cả các trường

BULLET_PATTERNS = {
    "ring":             BulletConfig(num_bullets=24, speed=DEFAULT_BULLET_SPEED, delay=1200, radius=DEFAULT_BULLET_RADIUS, color=GREEN),
    "rotating_ring":    BulletConfig(num_bullets=12, speed=DEFAULT_BULLET_SPEED, delay=2000, radius=DEFAULT_BULLET_RADIUS, color=YELLOW, rotation_speed=5),
    "bouncing":         BulletConfig(num_bullets=10, speed=DEFAULT_BULLET_SPEED, delay=2200, radius=DEFAULT_BULLET_RADIUS, color=BLUE),
    "spiral":           BulletConfig(num_bullets=36, speed=DEFAULT_BULLET_SPEED, delay=2500, radius=DEFAULT_BULLET_RADIUS, color=PURPLE, rotation_speed=5),
    "wave":             BulletConfig(num_bullets=10, speed=DEFAULT_BULLET_SPEED, delay=1800, radius=DEFAULT_BULLET_RADIUS, color=CYAN, wave_amplitude=30),
    "expanding_spiral": BulletConfig(num_bullets=36, speed=DEFAULT_BULLET_SPEED, delay=3000, radius=DEFAULT_BULLET_RADIUS, color=MAGENTA, speed_increment=0.1)
}