import tkinter as tk
from configs.game_config import DynamicConfig

def launch_configs_window():
    def update_value():
        try:
            DynamicConfig.DEFAULT_BULLET_SPEED = float(BULLET_SPEED_VAR.get())
            DynamicConfig.DISPLAY_BULLET_TRAIL = DISPLAY_BULLET_TRAIL_VAR.get()
            DynamicConfig.DISPLAY_PLAYER_TRAIL = DISPLAY_PLAYER_TRAIL_VAR.get()
            DynamicConfig.GAME_SPEED = float(GAME_SPEED_VAR.get())
            DynamicConfig.USE_BULLET_COLORS = USE_BULLET_COLORS_VAR.get()
        except ValueError:
            pass  #Ngăn chặn đầu vào lỗi
    
    window = tk.Tk()
    window.title("Game Config Panel")

    
    tk.Label(window, text = "Bullet Speed").pack()
    BULLET_SPEED_VAR = tk.StringVar(value = str(DynamicConfig.DEFAULT_BULLET_SPEED)) # khởi tạo giá trị hiện tại từ DynamicConfig
    tk.Entry(window, textvariable = BULLET_SPEED_VAR).pack() #tạo một ô nhập liệu để người dùng thay đổi giá trị

    DISPLAY_BULLET_TRAIL_VAR = tk.BooleanVar(value = DynamicConfig.DISPLAY_BULLET_TRAIL) #Tạo 1 biến kiểu bool liên kết với checkbox
    tk.Checkbutton(window,text = "Hiện vệt đạn", variable = DISPLAY_BULLET_TRAIL_VAR).pack() #Bật tắt checkbox

    tk.Label(window, text = "Game speed").pack()
    GAME_SPEED_VAR = tk.StringVar(value = str(DynamicConfig.GAME_SPEED))
    tk.Entry(window,textvariable = GAME_SPEED_VAR).pack()

    DISPLAY_PLAYER_TRAIL_VAR = tk.BooleanVar(value = DynamicConfig.DISPLAY_PLAYER_TRAIL) 
    tk.Checkbutton(window,text = "Hiện vệt người chơi", variable = DISPLAY_PLAYER_TRAIL_VAR).pack()

    USE_BULLET_COLORS_VAR = tk.BooleanVar(value = DynamicConfig.USE_BULLET_COLORS) #Tạo 1 biến kiểu bool liên kết với checkbox
    tk.Checkbutton(window,text = "Hiện vệt người chơi", variable = USE_BULLET_COLORS_VAR).pack()
    
    tk.Button(window, text="Apply", command = update_value).pack()
    window.mainloop()





