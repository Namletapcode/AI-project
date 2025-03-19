class Settings:
    def __init__ (self):
        self.screen_width = 1280
        self.screen_height = 640
        self.fps = 60
        self.dt_max = 3/self.fps
        self.player_speed = 2