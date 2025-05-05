from enum import Enum
from collections import namedtuple

SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
GAME_SPEED = 5.0                    # Tốc độ game (1.0 là tốc độ bình thường)
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

class DynamicConfig:
    DEFAULT_BULLET_SPEED = 150
    GAME_SPEED = 5.0
    DISPLAY_PLAYER_TRAIL = True
    DISPLAY_BULLET_TRAIL = False
    USE_BULLET_COLORS = False

DRAW_SECTOR_METHOD = DrawSectorMethod.USE_POLYGON

# Cấu hình vệt mờ
DISPLAY_PLAYER_TRAIL = False
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
    'speed',
    'spawn_time', # Tốc độ đạn (pixel/s) ở GAME_SPEED = 1.0
    'delay',
    'interval_delay',
    'delay_offset_limit',
    'init_delay',
    'radius',
    'color',
    'rotation_speed',  # cho rotating và spiral  # cho wave
    'count', # cho expanding
    'probability',
    'enable'
], defaults=[0] * 11)  # Set default 0 cho tất cả các trường

BULLET_PATTERNS = {
    "ring":     BulletConfig(num_bullets=24, speed=DEFAULT_BULLET_SPEED, delay=75,   interval_delay = 0, delay_offset_limit=20, radius=DEFAULT_BULLET_RADIUS, color=GREEN, probability=0.8, enable=True),
    "bouncing": BulletConfig(num_bullets=10, speed=DEFAULT_BULLET_SPEED, delay=100,  interval_delay = 0, delay_offset_limit=35, radius=DEFAULT_BULLET_RADIUS, color=BLUE, probability=0.5, enable=True),
    "spiral":   BulletConfig(num_bullets=36, speed=DEFAULT_BULLET_SPEED, delay=125,  interval_delay = 5, delay_offset_limit=50, radius=DEFAULT_BULLET_RADIUS, color=WHITE, rotation_speed=3, count=0, probability=0.8, enable=True),
    "tornado":  BulletConfig(num_bullets=6, speed=DEFAULT_BULLET_SPEED, delay=1000, interval_delay = 10, delay_offset_limit=250, radius=DEFAULT_BULLET_RADIUS, color=WHITE, rotation_speed=5, count=0, probability=1.0, enable=True),
    "sin_wave": BulletConfig(num_bullets=30, speed=DEFAULT_BULLET_SPEED, delay=75,   interval_delay = 5, delay_offset_limit=20, radius=DEFAULT_BULLET_RADIUS, color=WHITE, rotation_speed=5, count=16, probability=0.3, enable=False),
}