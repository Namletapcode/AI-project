import numpy as np
import cv2
import os
image_folder = "dataset/train"
image_list = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (256,256))
        image_list.append(gray_image.flatten())
        
        image_array = np.array(image_list)

np.save("dataset.npy", image_array)
data = np.load("dataset.npy")