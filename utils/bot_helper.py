import matplotlib.pyplot as plt

plt.ion()

def plot_training_progress(scores, mean_scores=None, title='Training...', window_size= 100):
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
    plt.legend()
    plt.show(block=False)
    plt.pause(0.05)

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

import cv2

def show_numpy_to_image(img: np.ndarray, img_size: int):
    # Show what the AI see
    vision = (img * 255).astype(np.uint8).reshape((img_size, img_size))
    cv2.imshow('AI Vision', vision)

# For testing visualization
if __name__ == '__main__':
    test_scores = []
    for game in range(100):
        score = game % 10 + (game // 10)
        test_scores.append(score)
        plot_training_progress(test_scores)