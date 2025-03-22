import pygame

class Bullet:
    def __init__(self, game, x, y):
        self.screen = game.screen
        self.settings = game.settings
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, 3, 10)


    def update(self):
        self.y -= 10
        self.rect.y = self.y

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.rect)