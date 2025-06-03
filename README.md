# Bullet Hell Game

Game né đạn 2D xây dựng bằng Pygame, AI tự động né đạn thông minh.

## Tổng Quan Dự Án

- Game engine xây dựng bằng Pygame
- Bot agent sử dụng cả phương pháp heuristic và deep learning
- Các chiến lược training model khác nhau
- Hệ thống đánh giá hiệu suất bot
- Công cụ trực quan hóa quá trình training và đánh giá

## Cấu Trúc Thư Mục

```
.
├── bot/               
│   ├── base_bot.py                       # Lớp bot cơ sở
│   ├── bot_manager.py                    # Quán quản lý các loại bot
│   └── heuristic_dodge.py                # Bot né đạn theo thuật toán (không dùng model/network)
│   └── deep_learning/                    # Bot dùng deep learning (đang phát triển)
│       ├── base_agent.py                 # Lớp agent cơ sở
│       ├── param_input/                  # Bot học từ input là vùng có đạn scan quanh player
│       │   ├── numpy_long_short_agent.py
│       │   ├── numpy_batch_interval_agent.py
│       │   ├── pytorch_agent.py
│       └── vision_input/                    # Bot học từ hình ảnh game
│           └── cupy_batch_interval_agent.py
│           └── cupy_long_short_agent.py
│           └── numpy_batch_interval_agent.py
│           └── numpy_long_short_agent.py
│           └── pytorch_agent.py
├── model/
│   ├── numpy_model.npy       # Model cài đặt bằng numpy
│   ├── cupy_model.npy        # Model cài đặt bằng cupy
│   └── torch_model.pth       # Model cài đặt bằng PyTorch
├── configs/
│   ├── bot_config.py         # Cấu hình cho bot
│   └── game_config.py        # Cấu hình game
├── game/              
│   ├── bullet.py             # Lớp đạn
│   ├── bullet_manager.py     # Quản lý đạn và mẫu đạn
│   ├── game_core.py          # Logic game chính
│   └── player.py             # Lớp người chơi
├── utils/
│   └── bot_helper.py         # Các hàm hỗ trợ cho bot
│   └── draw_utils.py         # Hỗ trợ vẽ ra màn hình game
│   └── interface.py          # Giao diện điều khiển
├── saved_files/
│   └── benchmark             # Chứa kết quả so sánh các bot sau train
│   └── param                 # Chứa các file model đã train, graph quá trình train 
│   └── vision                # Chứa các file model đã train, graph quá trình train 
│   └── supervised            # Chứa các file model đã train, graph quá trình train 
└── main.py                # Entry point của game
```

## Các loại bot

1. **Heuristic Dodge Bot**
   - Sử dụng các thuật toán dựa trên tham số game
   - Nhiều chiến lược né đạn:
      - Đường đi ít nguy hiểm nhất
      - Đường đi ít nguy hiểm nhất kết hợp dự đoán đạn
      - Hướng an toàn xa nhất
      - Hướng ngược với mối đe dọa
      - Vùng an toàn ngẫu nhiên

2. **Deep Learning Agent**
   - Sử dụng neural network để học cách né đạn
   - Hai phương pháp chính:
      - Học từ tham số game (parameter)
      - Học từ hình ảnh game (vision)
   - Hai chiến lược train model chính:
      1. **Long Short Memory**
         - Lưu trữ experience dài hạn và ngắn hạn
         - Training ngắn hạn: train mỗi bước (step-by-step)
         - Training dài hạn: train theo batch khi game over
         - Ưu điểm:
            - Học liên tục từ các state mới
            - Củng cố kiến thức qua training batch
            - Tăng tốc độ học và ổn định model
         - Nhược điểm: 
            - Chi phí tính toán cao do train hai lần
      
      2. **Batch Interval**
         - Lưu trữ liên tục experience vào replay buffer
         - Sau mỗi interval (khoảng thời gian):
            - Lấy ngẫu nhiên một batch từ replay buffer
            - Train model trên batch được sample
         - Ưu điểm:
            - Tránh tương quan giữa các experience liên tiếp
            - Training ổn định và hiệu quả
            - Tận dụng được GPU để train parallel
         - Nhược điểm:
            - Có thể miss các experience quan trọng khi sampling
            - Cần nhiều bộ nhớ để lưu replay buffer
   - Có 2 mode: training và perform
   - Lưu trữ experience để training
   - Benchmark hỗ trợ đánh giá so sánh, giao diện trực quan

## Cài đặt và Chạy

1. Clone repository:
```bash
git clone https://github.com/yourusername/BulletHellGame.git
cd BulletHellGame
```
2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
3. Thay đổi các thông số trong main.py và chạy
4. Chạy benchmark các bot đã huấn luyên:
```bash
python -m bot/evaluation/mark_Runner.py
```