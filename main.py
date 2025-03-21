import pygame
import sys
import time
from bullet_manager import BulletManager
from player import Player
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, dt_max
import math

class Game:
    def __init__(self):
        pygame.init()
        self.bullet_manager = BulletManager()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) 
        pygame.display.set_caption("Touhou")
        self.screen_rect = self.screen.get_rect()
        self.clock = pygame.time.Clock() 
        self.player = Player(self)
        self.group = pygame.sprite.Group()
        self.enemy_x, self.enemy_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        self.hit = False
        self.frame_index = 0
    
    def run(self):
        while True:
            self.dt = min(self.clock.tick(FPS) / 1000, dt_max)
            self.check_events()
            self.update_screen()
        

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


    def restart_game(self):
        self.__init__()
    def get_all_bullets_info(self):
        return [(bullet.x, bullet.y, math.degrees(bullet.angle)) for bullet in self.bullet_manager.bullets]
    
    def highlight_bullets_in_radius(self, d):
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.player.x), int(self.player.y)), d, 1)
        
        bullets_in_radius = []
        for bullet in self.bullet_manager.bullets:
            distance = math.sqrt((self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2)
            if distance <= d:
                bullet.set_color((255, 255, 0))  # Đổi màu đạn thành vàng
                bullets_in_radius.append(bullet)
        
        return bullets_in_radius

    def highlight_bullets_in_radius(self, d):
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.player.x), int(self.player.y)), d, 1)
        
        bullets_in_radius = []
        for bullet in self.bullet_manager.bullets:
            distance = math.sqrt((self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2)
            if distance <= d:
                bullet.color = (255, 255, 0)  # Đổi màu đạn thành vàng
                bullets_in_radius.append(bullet)
        
        return bullets_in_radius

    def update_screen(self):
        self.screen.fill((0, 0, 0))
        box_x, box_y, box_size = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 500
        pygame.draw.rect(self.screen, (255, 255, 255), 
                     (box_x - box_size // 2, box_y - box_size // 2, box_size, box_size), 2)
        self.player.update_player()
        self.player.draw_player()
        self.player.handle_screen_collision()
        
        if pygame.time.get_ticks() % 100 == 0:
            self.bullet_manager.create_ring()
        if pygame.time.get_ticks() % 151 == 0:
            self.bullet_manager.create_rotating_ring()
        if pygame.time.get_ticks() % 199 == 0:
            self.bullet_manager.create_spiral()
        if pygame.time.get_ticks() % 237 == 0:
            self.bullet_manager.create_wave()
        #if pygame.time.get_ticks() % 300 == 0:
           # self.bullet_manager.create_negative_speed_spiral(num_bullets=36, speed=-3, rotation_speed=5)
        if pygame.time.get_ticks() % 367 == 0:
            self.bullet_manager.create_expanding_spiral()
        if pygame.time.get_ticks() % 403 == 0:
            self.bullet_manager.create_bouncing_bullets()
        if pygame.time.get_ticks() % 450 == 0:
            self.bullet_manager.create_spiral_from_corners()
        if pygame.time.get_ticks() % 100 == 0:
            self.bullet_manager.create_targeted_shot(self.player.x, self.player.y, speed=4)
        
        self.bullet_manager.update()
        
        bullets_near_player = self.highlight_bullets_in_radius(100)  # Ví dụ dùng bán kính 100
        # print(len(bullets_near_player))
        
        self.bullet_manager.draw(self.screen)
        self.check_collision()
        self.group.update(self.dt)
        self.group.draw(self.screen)

        pygame.display.flip()
    def show_game_over_screen(self):
        font = pygame.font.Font(None, 74)
        text = font.render("Game Over", True, (255, 0, 0))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

        self.screen.fill((0, 0, 0))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

        time.sleep(2)  # Dừng game trong 2 giây
        self.restart_game()
    def check_collision(self):
        for bullet in self.bullet_manager.bullets:
            distance = math.sqrt((self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2)
            if distance <= self.player.radius + bullet.radius:
                self.show_game_over_screen()
game = Game()
game.run()