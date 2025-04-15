from enum import Enum

class DodgeAlgorithm(Enum):
    # Heuristic algorithms
    FURTHEST_SAFE_DIRECTION = 0
    LEAST_DANGER_PATH = 1
    LEAST_DANGER_PATH_ADVANCED = 2
    RANDOM_SAFE_ZONE = 3
    OPPOSITE_THREAT_DIRECTION = 4
    
    # Deep learning algorithms (placeholder for future)
    DQN = 5
    PPO = 6
    A2C = 7

# Dictionary mapping algorithms to their categories
ALGORITHM_CATEGORIES = {
    DodgeAlgorithm.FURTHEST_SAFE_DIRECTION: 'heuristic',
    DodgeAlgorithm.LEAST_DANGER_PATH: 'heuristic',
    DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED: 'heuristic',
    DodgeAlgorithm.RANDOM_SAFE_ZONE: 'heuristic',
    DodgeAlgorithm.OPPOSITE_THREAT_DIRECTION: 'heuristic',
    DodgeAlgorithm.DQN: 'deep_learning',
    DodgeAlgorithm.PPO: 'deep_learning',
    DodgeAlgorithm.A2C: 'deep_learning'
}

BOT_ACTION = True               # True if bot is allowed to take action : set by dev
BOT_DRAW = True                # True if bot is allowed to draw : set by dev
FILTER_MOVE_INTO_WALL = True
WALL_CLOSE_RANGE = 10

USE_WALL_PENALTY = True         # Phạt khi gần tường
# Mức độ ảnh hưởng của penalty
WALL_PENALTY_BIAS = 0.01        # Hệ số ảnh hưởng (càng cao, bot càng "ghét tường")
WALL_MARGIN = 30                # Khoảng cách từ tường bắt đầu tính penalty (ví dụ: dưới 50px tính là gần tường)

USE_COMPLEX_SCANNING = True
SCAN_RADIUS = 100
DODGE_ALGORITHM = DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED