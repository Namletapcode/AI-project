import pygame
from settings import Settings
from bullet import Bullet

class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        self.settings = Settings()
        self.screen = game.screen   
        self.screen_rect = game.screen_rect
        self.rect = pygame.Rect(self.settings.screen_width // 2, self.settings.screen_height - 40, 30, 30)
        self.direction = pygame.math.Vector2(0, 0)
        self.bullet = []  # Fixed: Keep this consistent

    def draw_player(self):
        pygame.draw.rect(self.screen, (255, 0, 0), self.rect)

    def update_player(self):
        self.input()
        self.move()
        for bullet in self.bullet[:]:  # Fixed loop
            bullet.update()
            bullet.draw()
            if bullet.rect.y < 0:  # Fixed: Use rect.y instead of just y
                self.bullet.remove(bullet)

    def input(self): 
        keys = pygame.key.get_pressed()
        self.direction.x = 0
        self.direction.y = 0

        if keys[pygame.K_LEFT]:
            self.direction.x = -1
        elif keys[pygame.K_RIGHT]:
            self.direction.x = 1

        if keys[pygame.K_UP]:
            self.direction.y = -1
        elif keys[pygame.K_DOWN]:
            self.direction.y = 1 

    def move(self):
        if self.direction.x and self.direction.y:
            self.direction = self.direction.normalize()
        self.rect.x += self.direction.x * self.settings.player_speed
        self.rect.y += self.direction.y * self.settings.player_speed

    def shoot(self):  # Fixed: Correct function definition
        bullet1 = Bullet(self, self.rect.x, self.rect.y)  # Fixed: Remove `self` from Bullet()
        bullet2 = Bullet(self, self.rect.x + 10, self.rect.y)
        bullet3 = Bullet(self, self.rect.x + 20, self.rect.y)
        
        self.bullet.append(bullet1)
        self.bullet.append(bullet2)    
        self.bullet.append(bullet3)
