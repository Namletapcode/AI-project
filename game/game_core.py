import pygame
import sys
import math
import numpy as np
from configs.game_config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, UPS,
    dt_max, BOX_LEFT, BOX_TOP, BOX_SIZE
)
from configs.bot_config import USE_COMPLEX_SCANNING, SCAN_RADIUS, IMG_SIZE
from utils.bot_helper import get_screen_shot_blue_channel, show_numpy_to_image
from game.bullet_manager import BulletManager
from game.player import Player

class Game:
    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Touhou")
        self.clock = pygame.time.Clock() 
        self.screen_rect = self.surface.get_rect()
        self.update_counter = 0
        self.player = Player(self.surface)
        self.bullet_manager = BulletManager(self.player)
        self.restart_game()
        self.font=pygame.font.Font(None, 36)
    
    def run(self, bot_manager, mode: str = "perform", render: bool = True, show_graph: bool = False):
        update_timer = 0
        update_interval = 1.0 / UPS
        first_frame = True
        if mode == "perform":
            if not bot_manager.is_heuristic:
                bot_manager.current_bot.set_mode("perform")
                try:
                    bot_manager.current_bot.load_model()
                except FileNotFoundError as e:
                    print(f"[ERROR] {e}")
                    pygame.quit()
                    sys.exit(1)
            while True:
                frame_time = min(self.clock.tick(FPS) / 1000, dt_max)
                update_timer += frame_time
                # Use first_frame to update immediately (to avoid not being able to update before drawing)
                while update_timer >= update_interval or first_frame:
                    current_state = self.get_state(bot_manager.is_heuristic, bot_manager.is_vision, bot_manager.is_numpy)
                    action = bot_manager.current_bot.get_action(current_state)
                    self.update(np.argmax(action))
                    if self.score in [250, 251, 252, 253]:
                        with open("log.txt", "a") as f:
                            f.write(f"Time: {self.update_counter},\nPlayer: ({self.player.x}, {self.player.y}),\nState: {[idx for idx, val in enumerate(current_state) if val == 1]},\nAction: {action}\n")
                    update_timer -= update_interval
                    first_frame = False
                if render:
                    self.draw(bot_manager.draw_bot_vision, current_state)
        else:
            bot_manager.current_bot.train(render, show_graph)
            
    def take_action(self, action_idx: int, render: bool = True): # for AI agent
        self.update(action_idx)
        if render:
            self.draw()

    def get_state(self, is_heuristic: bool = False, is_vision: bool = False, is_numpy: bool = True):
        """
        Get current game state as numpy array.
        
        Returns:
            np.ndarray: Array with bullet and wall information:
            - First N elements (N=8 or N=24): Indicate bullet presence in each region
              - If USE_COMPLEX_SCANNING=False: 8 elements for 8 directions
              - If USE_COMPLEX_SCANNING=True: 24 elements (8 directions x 3 distance rings)
              - Value 1 means bullet present, 0 means no bullet
            - Last 4 elements: Wall proximity flags [top, right, bottom, left]
              - Value 1 means near wall, 0 means not near wall
        """
        if is_heuristic:
            state = self.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
        else:
            if is_vision:
                if not hasattr(self, "img_01") or self.img_01 is None:
                    self.img_01 = np.zeros((IMG_SIZE ** 2, 1), dtype=np.float64)
                img_02 = get_screen_shot_blue_channel(self.player.x, self.player.y, IMG_SIZE, self.surface)
                state = np.concatenate((self.img_01, img_02), axis=0)
                self.img_01 = img_02
            else:
                bullets_in_radius = self.bullet_manager.get_bullet_in_range(SCAN_RADIUS)
                if USE_COMPLEX_SCANNING:
                    sector_flags = self.bullet_manager.get_complex_regions(bullets_in_radius)
                else:
                    sector_flags = self.bullet_manager.get_simple_regions(bullets_in_radius)
                near_wall_info = self.player.get_near_wall_info()
                
                # Combine states into single array
                state = np.zeros(len(sector_flags) + len(near_wall_info), dtype=np.float64)
                state[:len(sector_flags)] = sector_flags
                state[len(sector_flags):] = near_wall_info
            if is_numpy:
                state = state.reshape(len(state), 1)
        return state
    
    def get_reward(self) -> tuple[float, bool]:
        return self.reward if not self.game_over else -100.0, self.game_over

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def restart_game(self):
        self.player.reset()
        self.bullet_manager.reset(self.update_counter)
        self.reward = 0.5
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()

    def update(self, action_idx: int = None):
        self.update_counter += 1

        # update logic
        self.check_events()
        if not self.game_over:
            self.player.update(action_idx)
            self.reward = 0.5 if not self.player.is_moving else 0.0 # reset every loop, only set to zero if move, -10 if got hit
            if self.bullet_manager.key == 0:
                self.bullet_manager.update(update_num=self.update_counter)
            self.check_collision()
            self.score += 1
            self.survival_time = (pygame.time.get_ticks() - self.start_time) // 1000
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:
                self.restart_game()

    def draw(self, draw_extra: callable = None, current_state: np.ndarray = None):
        # re-draw surface
        self.surface.fill((0, 0, 0))
        self.draw_box()
        
        if draw_extra:
            draw_extra(current_state)
            
        self.player.draw()
        self.bullet_manager.draw(self.surface)
        # print(self.get_reward())
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        # time_text =  self.font.render(f"Time: {self.survival_time}s", True, (255, 255, 255))
    
        self.surface.blit(score_text, (10, 10))
        # self.surface.blit(time_text, (10, 40))
        pygame.display.flip()

    def draw_box(self):
        pygame.draw.rect(self.surface, (255, 255, 255), (BOX_TOP, BOX_LEFT, BOX_SIZE, BOX_SIZE), 2)

    def show_game_over_screen(self):
        text = self.font.render("Game Over", True, (255, 0, 0))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

        self.surface.fill((0, 0, 0))
        self.surface.blit(text, text_rect)
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