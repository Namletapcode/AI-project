import matplotlib.pyplot as plt
import pygame, os

plt.ion()  # Bật chế độ tương tác

def plot_training_progress(
    scores,
    mean_scores=None,
    title='Training...',
    window_size= 100,
    headless=False,
    save_dir=""
):
    """
    Plot training progress in real-time
    
    Args:
        scores: List of scores to plot
        title: Title of the plot
    """
    def moving_average(data, window):
        """
        Computes the moving average over a list of values using a fixed window size.

        Parameters:
        - data (list or np.ndarray): The sequence of values (e.g., episode rewards).
        - window (int): The size of the moving window.

        Returns:
        - np.ndarray: The smoothed values using a simple moving average.

        How it works:
        - np.ones(window) creates an array of ones, e.g., np.ones(3) → [1, 1, 1]
        - Dividing by window gives equal weights: np.ones(3)/3 → [1/3, 1/3, 1/3]
        This acts as an averaging filter (called a "kernel").
        - np.convolve(data, kernel, mode='valid') slides the kernel across the data,
        computing the average at each valid position (i.e., where the full window fits).
        """
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.clf()
    plt.plot(scores, label='Score')
    # plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores is None:
        if len(scores) >= window_size:
            avg_scores = moving_average(scores, window_size)
            plt.plot(range(window_size - 1, len(scores)), avg_scores, label=f'Average score', linewidth=2)
    else:
        plt.plot(mean_scores, label='Average Score')
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.legend() # Show annotate
    
    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True) # Create folder
        plt.savefig(save_dir)
    if not headless:
        plt.pause(0.05)


import sys
if sys.platform == "win32":
    import win32gui
    import win32ui
    import win32con
import numpy as np

def get_screen_shot_gray_scale(x: float, y: float, img_size: int) -> np.ndarray:
    """
    Params:
        x: horizontal position of player
        y: vertical position of player
        img_size: the size (pixels) of the square screen shot will be taken

    Return:
        a numpy array of the gray scale of each every pixels in the region taken with the size of (img_size ** 2, 1)

    Warning:
        Make sure that input satisfy x > img_size / 2 and y > img_size / 2
    """
    x = int(x - img_size / 2)
    y = int(y - img_size / 2)

    windowname = "Touhou"
    hwnd = win32gui.FindWindow(None, windowname)

    # Capture only the client area
    wDC = win32gui.GetDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, img_size, img_size)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (img_size, img_size), dcObj, (x, y), win32con.SRCCOPY)
    
    # Convert bitmap to numpy.array
    bmp_bytes = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmp_bytes, dtype=np.uint8).reshape((img_size * img_size, 4))

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    # Convert to grayscale (normalize to 0-1)
    # BGRX format: [:, 0]=Blue, [:, 1]=Green, [:, 2]=Red, [:, 3]=Unused
    r = img[:, 2].astype(np.float64)
    g = img[:, 1].astype(np.float64)
    b = img[:, 0].astype(np.float64)

    result = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    result = result.reshape((-1, 1))  # shape: (img_size * img_size, 1)

    return result

def get_screen_shot_blue_channel(x: float, y: float, img_size: int, surface: pygame.Surface) -> np.ndarray:
    """
    Lấy ảnh grayscale từ blue channel của Pygame surface.
    
    Args:
        x (float): Tọa độ x tâm ảnh.
        y (float): Tọa độ y tâm ảnh.
        img_size (int): Kích thước ảnh (ảnh sẽ là hình vuông).
        surface (pygame.Surface): Surface nguồn để chụp ảnh.

    Returns:
        np.ndarray: Ảnh grayscale từ kênh blue, shape = (img_size, img_size), dtype=float32, range [0, 1].
    """
    # Trích kênh Blue từ surface (không .copy vì chỉ đọc)
    blue_channel = pygame.surfarray.pixels_blue(surface)  # shape (width, height)

    # Chuyển từ (W, H) sang (H, W) để đúng với xử lý ảnh
    blue_channel = np.transpose(blue_channel)
    
    # Cắt ảnh xung quanh điểm (x, y)
    half = img_size // 2
    x, y = int(x), int(y)
    
     # Tính chỉ số cắt
    top = max(0, y - half)
    bottom = min(blue_channel.shape[0], y + half)
    left = max(0, x - half)
    right = min(blue_channel.shape[1], x + half)
    
    # Tạo ảnh kết quả có đúng shape (img_size, img_size)
    cropped = np.zeros((img_size, img_size), dtype=np.float32)
    crop_data = blue_channel[top:bottom, left:right]
    
    # Dán phần lấy được vào giữa ảnh đầu ra
    h, w = crop_data.shape
    cropped[:h, :w] = crop_data
    
    # Normalize và flatten về 1 chiều
    return (cropped / 255.0).reshape(-1, 1)  # shape: (img_size*img_size, 1)

import cv2

def show_numpy_to_image(img: np.ndarray, img_size: int):
    # Show what the AI see
    vision = (img * 255).astype(np.uint8).reshape((img_size, img_size))
    cv2.imshow('AI Vision', vision)
    cv2.waitKey(1)

# For testing visualization
if __name__ == '__main__':
    test_scores = []
    for game in range(100):
        score = game % 10 + (game // 10)
        test_scores.append(score)
        plot_training_progress(test_scores)
