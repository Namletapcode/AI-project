import pygame
from configs.game_config import *
from options_menu import *

class Menu:
    def __init__(self,screen):
        self.screen = screen
        self.font=pygame.font.Font(None, 36)
        self.options = ["Playing", "Options", "Quit"]
        self.selected_index = 0
    
    def draw(self):
        self.screen.fill((0, 0, 0))
        title_text = self.font.render("Touhou Mini Game", True, (255, 255, 255))
        self.screen.blit(title_text, (self.screen.get_width() // 2 - title_text.get_width() // 2, 100))

        for i, option in enumerate(self.options):   #dòng dc chọn thì chữ vàng còn lại đổi sang chữ trắng
            color = (255, 255, 0) if i == self.selected_index else (255, 255, 255)
            text = self.font.render(option, True, color)
            self.screen.blit(text, (self.screen.get_width() // 2 - text.get_width() // 2, 200 + i * 40))

        pygame.display.flip()
    
    def handle_input(self,event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index-1) % len(self.options)
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index+1) % len(self.options)
            elif event.key == pygame.K_RETURN:
                return self.options[self.selected_index]
        return None