import os
import numpy as np
import cv2

def scan_image(image, rgb_tuple):
    # Convert RGB tuple to numpy array for comparison
    rgb_array = np.array(rgb_tuple)
    image = np.array(image)
    shape = image.shape
    mask = np.zeros(shape[:2],dtype="uint8")
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image[i,j,0] == rgb_tuple[0] and image[i,j,1] == rgb_tuple[1] and image[i,j,2] == rgb_tuple[2]:
                mask[i,j] = 1

    return mask

