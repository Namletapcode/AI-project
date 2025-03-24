from enum import Enum

SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
FPS = 60
dt_max = 3 / FPS
PLAYER_SPEED = 3
DEFAULT_BULLET_SPEED = 2.5
DEFAULT_BULLET_RADIUS = 5
BOX_SIZE = 500
BOX_TOP = (SCREEN_HEIGHT - BOX_SIZE) / 2
BOX_LEFT = (SCREEN_WIDTH - BOX_SIZE) / 2

class DodgeMethod(Enum):
    LEAST_DANGER_PATH = 0
    FURTHEST_SAFE_DIRECTION = 1
    RANDOM_SAFE_ZONE = 2
    OPPOSITE_THREAT_DIRECTION = 3

class DrawSectorMethod(Enum):
    USE_POLYGON = 0
    USE_TRIANGLE = 1
    USE_TRIANGLE_AND_ARC = 2
    USE_PIL = 3

USE_BOT = True
SCAN_RADIUS = 100
DODGE_METHOD = DodgeMethod.OPPOSITE_THREAT_DIRECTION
DRAW_SECTOR_METHOD = DrawSectorMethod.USE_POLYGON

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

class BulletBase:
    """Lớp cơ sở cho tất cả loại đạn."""
    def __init__(self, num_bullets, speed, radius=DEFAULT_BULLET_RADIUS, color=WHITE, fade=0):
        self.num_bullets = num_bullets
        self.speed = speed
        self.radius = radius
        self.color = color
        self.fade = fade

class RingBullet(BulletBase):
    """Đạn vòng tròn."""
    def __init__(self, num_bullets=24, speed=DEFAULT_BULLET_SPEED, radius=DEFAULT_BULLET_RADIUS, color=GREEN):
        super().__init__(num_bullets, speed, radius, color)

class BouncingBullet(BulletBase):
    """Đạn nảy."""
    def __init__(self, num_bullets=10, speed=DEFAULT_BULLET_SPEED, radius=DEFAULT_BULLET_RADIUS, color=BLUE):
        super().__init__(num_bullets, speed, radius, color)
        
class RotatingRingBullet(BulletBase):
    """Đạn vòng tròn quay."""
    def __init__(self, num_bullets=12, speed=DEFAULT_BULLET_SPEED, rotation_speed=5, radius=DEFAULT_BULLET_RADIUS, color=YELLOW):
        super().__init__(num_bullets, speed, radius, color)
        self.rotation_speed = rotation_speed

class SpiralBullet(BulletBase):
    """Đạn xoắn ốc."""
    def __init__(self, num_bullets=36, speed=DEFAULT_BULLET_SPEED, rotation_speed=5, radius=DEFAULT_BULLET_RADIUS, color=PURPLE):
        super().__init__(num_bullets, speed, radius, color)
        self.rotation_speed = rotation_speed

class WaveBullet(BulletBase):
    """Đạn dạng sóng."""
    def __init__(self, num_bullets=10, speed=DEFAULT_BULLET_SPEED, wave_amplitude=30, radius=DEFAULT_BULLET_RADIUS, color=CYAN):
        super().__init__(num_bullets, speed, radius, color)
        self.wave_amplitude = wave_amplitude

class ExpandingSpiralBullet(BulletBase):
    """Đạn xoắn ốc mở rộng."""
    def __init__(self, num_bullets=36, speed=DEFAULT_BULLET_SPEED, speed_increment=0.1, radius=DEFAULT_BULLET_RADIUS, color=MAGENTA):
        super().__init__(num_bullets, speed, radius, color)
        self.speed_increment = speed_increment