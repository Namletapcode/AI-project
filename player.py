import pygame
from settings import Settings
from bullet import Bullet
import math
class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__() #kế thừa lớp con từ lớp cha
        self.settings=Settings()
        self.screen=game.screen
        self.screen_rect=game.screen_rect
        #self.rect=pygame.Rect(self.settings.screen_width//2,self.settings.screen_height-40,30,30)
        self.direction=pygame.Vector2(0,0)
        self.bullets=[]
        self.radius=5
        self.x=self.settings.screen_width//2
        self.y=self.settings.screen_height-100
        
    def draw_player(self):
        pygame.draw.circle(self.screen,(255,0,0),(self.x,self.y),self.radius)
    def update_player(self):
        self.input()
        self.move()
        for bullet in self.bullets:
           bullet.update()
           bullet.draw()
           if bullet.y <0:
               self.bullets.remove(bullet)

    def move_left(self):
        self.x -= self.settings.player_speed
        self.handle_screen_collision()
    def move_right(self):
        self.x += self.settings.player_speed
        self.handle_screen_collision
    def move_up(self):
        self.y -= self.settings.player_speed
        self.handle_screen_collision
    def move_down(self):
        self.y += self.settings.player_speed
        self.handle_screen_collision()
     
    def input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.x -= self.settings.player_speed
        if keys[pygame.K_RIGHT]:
            self.x += self.settings.player_speed
        if keys[pygame.K_UP]:
            self.y -= self.settings.player_speed
        if keys[pygame.K_DOWN]:
            self.y += self.settings.player_speed
    def move_diagonal_up_left(self):
      self.x -= self.settings.player_speed
      self.y -= self.settings.player_speed
      self.handle_screen_collision()

def move_diagonal_up_right(self):
    self.x += self.settings.player_speed
    self.y -= self.settings.player_speed
    self.handle_screen_collision()

def move_diagonal_down_left(self):
    self.x -= self.settings.player_speed
    self.y += self.settings.player_speed
    self.handle_screen_collision()

def move_diagonal_down_right(self):
    self.x += self.settings.player_speed
    self.y += self.settings.player_speed
    self.handle_screen_collision()
  
    def move(self):
         if self.direction.x and self.direction.y:
             self.direction.normalize()
         self.x+=self.direction.x * self.settings.player_speed
         self.y+=self.direction.y * self.settings.player_speed
    def handle_screen_collision(self):
        """Ngăn hình tròn đi ra ngoài màn hình"""
        if self.x - self.radius < 0:
            self.x = self.radius  # Giữ trong giới hạn trái
        if self.x + self.radius > self.settings.screen_width:
            self.x = self.settings.screen_width - self.radius  # Giữ trong giới hạn phải
        if self.y - self.radius < 0:
            self.y = self.radius  # Giữ trong giới hạn trên
        if self.y + self.radius > self.settings.screen_height:
            self.y = self.settings.screen_height - self.radius
        box_x, box_y, box_size = self.settings.screen_width // 2, self.settings.screen_height // 2, 500
        left = box_x - box_size // 2
        right = box_x + box_size // 2
        top = box_y - box_size // 2
        bottom = box_y + box_size // 2

        if self.x - self.radius < left:
           self.x = left + self.radius  
        if self.x + self.radius > right:
           self.x = right - self.radius  
        if self.y - self.radius < top:
           self.y = top + self.radius  
        if self.y + self.radius > bottom:
           self.y = bottom - self.radius  
   
        
