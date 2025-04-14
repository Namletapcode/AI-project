from abc import ABC, abstractmethod

class BaseBot(ABC):
    def __init__(self, game):
        self.game = game
        self.player = game.player

    @abstractmethod
    def get_action(self) -> int:
        """Tính toán và trả về index của hướng di chuyển tốt nhất."""
        pass

    def draw(self):
        """Vẽ debug, hiển thị tầm nhìn, … nếu cần."""
        pass