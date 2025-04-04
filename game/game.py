import pygame
import sys
import math
import threading
import numpy as np
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, FPS, dt_max, BOX_LEFT, BOX_TOP, BOX_SIZE, DODGE_METHOD, BOT_DRAW)
from bullet_manager import BulletManager
from player import Player
from bot_ai import GameBot

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Touhou")
        self.clock = pygame.time.Clock() 
        self.screen_rect = self.screen.get_rect()
        self.restart_game()
        self.font=pygame.font.Font(None, 36)
    
    def run(self):
        while True:
            self.dt = min(self.clock.tick(FPS) / 1000, dt_max)
            self.update_screen()

    def take_action(self, action: np.ndarray): # for AI agent
        self.dt = min(self.clock.tick(FPS) / 1000, dt_max)
        self.update(action)
        self.draw()

    def get_state(self) -> np.ndarray:
        return self.bullet_manager.get_state()
    
    def get_reward(self) -> tuple[float, bool]:
        return self.reward if not self.game_over else -10, self.game_over

    def run_in_another_thread(self):    # tạm thời không di chuyển được nếu chạy song luồng
        new_thread = threading.Thread(target=self.run, daemon=True)
        new_thread.start()
        new_thread.join()

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            self.bullet_manager.spawn_random_bullet_pattern(event)

    def restart_game(self):
        self.player = Player(self)
        self.bullet_manager = BulletManager(self.player)
        self.enemy_x, self.enemy_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        self.frame_index = 0
        self.reward = 0.1
        self.bot = GameBot(self, DODGE_METHOD)
        self.game_over = False
        self.score=0
        self.start_time=pygame.time.get_ticks()

    def update(self, action: np.ndarray = None):
        # update logic
        self.check_events()
        if not self.game_over:
            self.reward = 0.1 # reset every loop, only set to zero if move, -10 if got hit
            self.bot.update()
            self.player.update(action)
            self.bullet_manager.update()
            self.check_collision()
            self.score+=1
            self.survival_time=(pygame.time.get_ticks()-self.start_time) // 1000
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:
                self.game_over = False
                self.restart_game()

    def draw(self):
        # re-draw screen
        self.screen.fill((0, 0, 0))
        self.draw_box()
        if BOT_DRAW:
            self.bot.draw_vison()
        self.player.draw()
        self.bullet_manager.draw(self.screen)
        # print(self.get_reward())
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        time_text =  self.font.render(f"Time: {self.survival_time}s", True, (255, 255, 255))
    
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 40))
        pygame.display.flip()
    def draw_box(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (BOX_TOP, BOX_LEFT, BOX_SIZE, BOX_SIZE), 2)

    def update_screen(self):
        # main user update funtion!
        self.update(self.bot.action) #if BOT_ACTION else None
        self.draw()

    def show_game_over_screen(self):
        text = self.font.render("Game Over", True, (255, 0, 0))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

        self.screen.fill((0, 0, 0))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

        # time.sleep(2)  # Dừng game trong 2 giây
        self.restart_game()

    def check_collision(self):
        # if colision restart game
        for bullet in self.bullet_manager.bullets:
            distance = math.sqrt((self.player.x - bullet.x) ** 2 + (self.player.y - bullet.y) ** 2)
            if distance <= self.player.radius + bullet.radius:
                self.game_over = True
                break