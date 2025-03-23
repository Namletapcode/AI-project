SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
FPS = 60
dt_max = 3 / FPS
PLAYER_SPEED = 3
DEFAULT_BULLET_SPEED = 2.5
DEFAULT_BULLET_RADIUS = 5

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

WHITE = BLACK = RED = GREEN = BLUE = YELLOW = PURPLE = (255, 255, 255)

class BulletBase:
    """Lớp cơ sở cho tất cả loại đạn."""
    def __init__(self, num_bullets, speed, color=WHITE, fade=0):
        self.num_bullets = num_bullets
        self.speed = speed
        self.color = color
        self.fade = fade

class RingBullet(BulletBase):
    """Đạn vòng tròn."""
    def __init__(self, num_bullets=24, speed=DEFAULT_BULLET_SPEED):
        super().__init__(num_bullets, speed, color=GREEN)

class BouncingBullet(BulletBase):
    """Đạn nảy."""
    def __init__(self, num_bullets=10, speed=DEFAULT_BULLET_SPEED):
        super().__init__(num_bullets, speed, color=BLUE)
        
class RotatingRingBullet(BulletBase):
    """Đạn vòng tròn quay."""
    def __init__(self, num_bullets=12, speed=DEFAULT_BULLET_SPEED, rotation_speed=5):
        super().__init__(num_bullets, speed, color=YELLOW)
        self.rotation_speed = rotation_speed

class SpiralBullet(BulletBase):
    """Đạn xoắn ốc."""
    def __init__(self, num_bullets=36, speed=DEFAULT_BULLET_SPEED, rotation_speed=5):
        super().__init__(num_bullets, speed, color=PURPLE)
        self.rotation_speed = rotation_speed

class WaveBullet(BulletBase):
    """Đạn dạng sóng."""
    def __init__(self, num_bullets=10, speed=DEFAULT_BULLET_SPEED, wave_amplitude=30):
        super().__init__(num_bullets, speed)
        self.wave_amplitude = wave_amplitude

class ExpandingSpiralBullet(BulletBase):
    """Đạn xoắn ốc mở rộng."""
    def __init__(self, num_bullets=36, initial_speed=DEFAULT_BULLET_SPEED, speed_increment=0.1):
        super().__init__(num_bullets, initial_speed)
        self.initial_speed = initial_speed
        self.speed_increment = speed_increment