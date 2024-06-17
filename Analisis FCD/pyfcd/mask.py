import numpy as np
import cv2

def masked(i_def, i_ref, N=15, H=10, low=100, high=255):
    blurred_image = cv2.GaussianBlur(i_def, (N, N), H)

    _, binary = cv2.threshold(blurred_image, low, high, cv2.THRESH_BINARY)
    kernel = np.ones((N, N), np.uint8)
    mask_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)//255

    return i_def*mask_open + i_ref * (1-mask_open)
