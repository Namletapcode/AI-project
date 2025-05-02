import pygame
from configs.game_config import *

class Options_Menu:
    def __init__(self,screen,font):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font=pygame.font.Font(None, 36)
        self.options = [
            "Control Mode: AI",
            "Bullet Speed: 5",
            "Back"
            ]
        self.selected_index = 0
        self.control_mode = "AI"
        self.bullet_speed = 5
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        title_text = self.font.render("Options", True, (255, 255, 255))
        self.screen.blit(title_text, (self.screen.get_width() // 2 - title_text.get_width() // 2, 100))

        for i, option in enumerate(self.options):
            display_text = option
            if "Control Mode" in option:
                display_text = f"Control Mode: {self.control_mode}"
            elif "Bullet Speed" in option:
                display_text = f"Bullet Speed: {self.bullet_speed}"

            color = (255, 255, 0) if i == self.selected_index else (255, 255, 255)
            text = self.font.render(display_text, True, color)
            self.screen.blit(text, (self.screen.get_width() // 2 - text.get_width() // 2, 200 + i * 40))

        pygame.display.flip()

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
           if event.key == pygame.K_UP:
              self.selected_index = (self.selected_index - 1) % len(self.options)
           elif event.key == pygame.K_DOWN:
              self.selected_index = (self.selected_index + 1) % len(self.options)
           elif event.key == pygame.K_RETURN:
              if self.selected_index == 0:  # Toggle control mode
                 self.control_mode = "Player" if self.control_mode == "AI" else "AI"
              elif self.selected_index == 1:  # Change bullet speed
                 self.bullet_speed += 1
                 if self.bullet_speed > 10:
                    self.bullet_speed = 1
              elif self.selected_index == 2:  # Back
                 return "Back"
        return None