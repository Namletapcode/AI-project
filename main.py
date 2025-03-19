import sys
import pygame
import time
from settings import Settings
from player import Player
from enemy import BulletManager
import math

class Game:
    def __init__(self):
        pygame.init()
        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        pygame.display.set_caption('Touhou')
        self.clock = pygame.time.Clock()
        self.screen_rect = self.screen.get_rect()
        self.clock  = pygame.time.Clock()
        self.player = Player(self)
        self.group = pygame.sprite.Group()
        self.FIRE_BULLET_EVENT = pygame.USEREVENT + 1   
        pygame.time.set_timer(self.FIRE_BULLET_EVENT, 100)

        self.space_surface = pygame.image.load('Touhou/image/space3.gif').convert()
        self.enemy_x, self.enemy_y = self.settings.screen_width // 2, self.settings.screen_height // 2
        self.bullet_manager = BulletManager()

    def run(self):
        while True:
            self.dt = min(self.clock.tick(self.settings.fps) / 1000, self.settings.dt_max)
            self.check_events()
            self.update_screen()
        
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == self.FIRE_BULLET_EVENT:
                self.player.shoot()

    def get_all_bullets_info(self):
            bullet_info = [(bullet.x, bullet.y, math.degrees(bullet.angle)) for bullet in self.bullet_manager.bullets]
            return bullet_info
            
    def update_screen(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.space_surface, (0, 0))
        self.player.update_player()
        self.player.draw_player()
        if pygame.time.get_ticks() % 100 == 0:
            self.bullet_manager.create_ring(self.enemy_x, self.enemy_y, num_bullets=24, speed=3)
        if pygame.time.get_ticks() % 151 == 0:
            self.bullet_manager.create_rotating_ring(self.enemy_x, self.enemy_y, num_bullets=12, speed=3, rotation_speed=5)
        if pygame.time.get_ticks() % 199 == 0:    
            self.bullet_manager.create_spiral(self.enemy_x, self.enemy_y, num_bullets=36, speed=3, rotation_speed=5)
        if pygame.time.get_ticks() % 237 == 0:
            self.bullet_manager.create_wave(self.enemy_x, self.enemy_y, num_bullets=10, speed=3, wave_amplitude=30)
        #if pygame.time.get_ticks() % 130 == 0:
            #self.bullet_manager.create_half_ring(self.enemy_x, self.enemy_y, num_bullets=18, speed=3, aim=90, width=180)
        if pygame.time.get_ticks() % 300 == 0:
            self.bullet_manager.create_negative_speed_spiral(self.enemy_x, self.enemy_y, num_bullets=36, speed=-3, rotation_speed=5)
        if pygame.time.get_ticks() % 367 == 0:
            self.bullet_manager.create_expanding_spiral(self.enemy_x, self.enemy_y, num_bullets=36, initial_speed=2, speed_increment=0.1)
        if pygame.time.get_ticks() % 403 == 0:
            self.bullet_manager.create_bouncing_bullets(self.enemy_x, self.enemy_y, num_bullets=10, speed=4)
        if pygame.time.get_ticks() % 450 == 0:
            self.bullet_manager.create_spiral_from_corners(self.settings.screen_width, self.settings.screen_height, num_bullets=36, speed=3, rotation_speed=5)

        
        self.bullet_manager.update()
        self.bullet_manager.draw(self.screen)
        self.group.update(self.dt)
        self.group.draw(self.screen)
        pygame.display.flip()

        bullet_data = self.get_all_bullets_info()
        print(bullet_data)  # In ra danh sách tọa độ và góc của tất cả đạn

            
    


game = Game()
game.run()


