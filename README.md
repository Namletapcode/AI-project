# Bullet Hell Game

Game né đạn 2D được xây dựng bằng Pygame, với các mẫu đạn đa dạng và AI tự động né đạn.

## Cấu trúc thư mục hiện tại

```
.
├── agent_01/               # AI Agent sử dụng Deep Q-Learning (đang thử nghiệm)
│   ├── agent.py            # Agent chính với train/perform mode
│   └── model.py            # Neural network model
├── bot/               
│   ├── base_bot.py         # Lớp bot cơ sở
│   ├── bot_manager.py      # Factory pattern quản lý các loại bot
│   └── heuristic_dodge.py  # Bot né đạn theo thuật toán (không dùng model/network)
│   └── deep_learning/      # Bot dùng deep learning (đang phát triển)
│       ├── param_input/    # Bot học từ tham số game (vị trí đạn, tốc độ,...)
│       └── vision_input/   # Bot học từ hình ảnh game
├── game/              
│   ├── bullet.py      # Quản lý đạn
│   ├── game.py        # Logic game chính
│   └── player.py      # Quản lý người chơi
└── main.py            # Entry point của game
```

## Các loại bot

1. **Heuristic Dodge Bot**
   - Sử dụng thuật toán né đạn đơn giản dựa trên khoảng cách và hướng di chuyển
   - KHÔNG sử dụng model hay neural network
   - Tính toán vị trí an toàn dựa trên vị trí các viên đạn
   - Phù hợp làm baseline cho việc so sánh với các phương pháp AI

2. **Deep Q-Learning Agent**
   - Sử dụng neural network để học cách né đạn
   - Có 2 mode: training và perform
   - Lưu trữ experience để training
   - Có khả năng học và cải thiện qua thời gian