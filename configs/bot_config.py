from enum import Enum

class DodgeMethod(Enum):
    LEAST_DANGER_PATH = 0
    LEAST_DANGER_PATH_ADVANCED = 1
    FURTHEST_SAFE_DIRECTION = 2
    RANDOM_SAFE_ZONE = 3
    OPPOSITE_THREAT_DIRECTION = 4

BOT_ACTION = True # True if bot is allowed to take action : set by dev
BOT_DRAW = False # True if bot is allowed to draw : set by dev
FILTER_MOVE_INTO_WALL = True
WALL_CLOSE_RANGE = 10

USE_WALL_PENALTY = True # Phạt khi gần tường
# Mức độ ảnh hưởng của penalty
WALL_PENALTY_BIAS = 0.01 #Hệ số ảnh hưởng (càng cao, bot càng "ghét tường")
WALL_MARGIN = 30        # Khoảng cách từ tường bắt đầu tính penalty (ví dụ: dưới 50px tính là gần tường)

SCAN_RADIUS = 100
DODGE_METHOD = DodgeMethod.LEAST_DANGER_PATH_ADVANCED