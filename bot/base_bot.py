

class BaseBot():
    def __init__(self, game):
        self.game = game
        self.player = game.player

    def get_action(self) -> int:
        """
        Calculate and return best index direction
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def draw(self) -> None:
        """
        Draw bot vision, scan bullet, ...
        """
        raise NotImplementedError("This method should be implemented in subclasses.")