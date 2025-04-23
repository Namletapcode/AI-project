from enum import Enum

class DodgeAlgorithm(Enum):
    # Heuristic algorithms
    FURTHEST_SAFE_DIRECTION = 0
    LEAST_DANGER_PATH = 1
    LEAST_DANGER_PATH_ADVANCED = 2
    RANDOM_SAFE_ZONE = 3
    OPPOSITE_THREAT_DIRECTION = 4
    
    # Deep learning algorithms
    DL_PARAM_INPUT_NUMPY = 5
    DL_PARAM_INPUT_TORCH = 6
    DL_VISION_INPUT_NUMPY = 7
    DL_VISION_INPUT_TORCH = 8 

BOT_ACTION = True               # True if bot is allowed to take action : set by dev
BOT_DRAW = False                # True if bot is allowed to draw : set by dev
FILTER_MOVE_INTO_WALL = True
WALL_CLOSE_RANGE = 30

USE_WALL_PENALTY = True         # Phạt khi gần tường
# Mức độ ảnh hưởng của penalty
WALL_PENALTY_BIAS = 0.01        # Hệ số ảnh hưởng (càng cao, bot càng "ghét tường")
WALL_MARGIN = 30                # Khoảng cách từ tường bắt đầu tính penalty (ví dụ: dưới 50px tính là gần tường)

USE_COMPLEX_SCANNING = True
SCAN_RADIUS = 100
DODGE_ALGORITHM = DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED