import matplotlib.pyplot as plt
import win32gui
import win32ui
import win32con
import numpy as np
import time

plt.ion()

def plot(scores):
    plt.clf()  # Clear current figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.legend()
    plt.pause(0.05)  # Pause to allow the plot to update

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

# testing
if __name__ == '__main__':
    start = time.time()
    temp = get_screen_shot_gray_scale(650 / 2, 650 / 2, 100)
    end = time.time()
    print(temp.shape, "Time:", end - start)