from enum import Enum
from multiprocessing import Manager

class DodgeAlgorithm(Enum):
    # Heuristic algorithms
    FURTHEST_SAFE_DIRECTION = 0
    LEAST_DANGER_PATH = 1
    LEAST_DANGER_PATH_ADVANCED = 2
    RANDOM_SAFE_ZONE = 3
    OPPOSITE_THREAT_DIRECTION = 4
    
    # Deep learning algorithms
    DL_PARAM_BATCH_INTERVAL_NUMPY = 5
    DL_PARAM_LONG_SHORT_NUMPY = 6
    DL_PARAM_CUPY = 7
    DL_PARAM_TORCH = 8
    DL_VISION_BATCH_INTERVAL_NUMPY = 9
    DL_VISION_LONG_SHORT_NUMPY = 11
    DL_VISION_BATCH_INTERVAL_CUPY = 12
    DL_VISION_LONG_SHORT_CUPY = 13
    DL_VISION_TORCH = 14
    SUPERVISED = 15

class SharedState:
    def __init__(self):
        manager = Manager()
        self._shared_dict = manager.dict()
        self._shared_dict['bot_draw'] = False
        self._shared_dict['is_vision'] = False
        
    @property
    def bot_draw(self):
        return self._shared_dict['bot_draw']
    
    @property
    def is_vision(self):
        return self._shared_dict['is_vision']
    
    @bot_draw.setter
    def bot_draw(self, value):
        self._shared_dict['bot_draw'] = value
    
    @is_vision.setter
    def is_vision(self, value):
        self._shared_dict['is_vision'] = value

BOT_ACTION = True               # True if bot is allowed to take action : set by dev
bot_draw = False                # True if bot is allowed to draw : set by dev
FILTER_MOVE_INTO_WALL = True
WALL_CLOSE_RANGE = 30

USE_WALL_PENALTY = True         # Phạt khi gần tường
# Mức độ ảnh hưởng của penalty
WALL_PENALTY_BIAS = 0.01        # Hệ số ảnh hưởng (càng cao, bot càng "ghét tường")
WALL_MARGIN = 30                # Khoảng cách từ tường bắt đầu tính penalty (ví dụ: dưới 50px tính là gần tường)

USE_COMPLEX_SCANNING = True
SCAN_RADIUS = 100
DODGE_ALGORITHM = DodgeAlgorithm.LEAST_DANGER_PATH_ADVANCED

IMG_SIZE = 50
DATE_FORMAT = "%d-%m %H:%M:%S"