import matplotlib.pyplot as plt
import numpy as np
import cv2

def mask(image, N=15, H=10, low=80, high=255, show_mask=False):
    blurred_image = cv2.GaussianBlur(image, (N, N), H)    
    
    _, binary = cv2.threshold(blurred_image, low, high, cv2.THRESH_BINARY)
    kernel = np.ones((N, N), np.uint8)
    mask_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel).astype(np.uint8)

    if show_mask:
        plt.imshow(mask_open)
        plt.show()
    return mask_open  

def edges(image, low, high):
    return cv2.Canny(mask(image), low, high) 

def center(image, low=20, high=40, show_center=False): 
    image = image.astype(np.uint8)
    circles = cv2.HoughCircles(edges(image, low, high), cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=40, param2=20, minRadius=40, maxRadius=80)
    c = (int(circles[0][0, 0]), int(circles[0][0, 1])) if circles is not None else None
    if show_center and c:
        plt.imshow(image)
        plt.scatter([c[0]], [c[1]])
        plt.show()
    return c  

def update_roi_center(old_roi, new_center):
    x_old, y_old, w, h = old_roi
    cx_local, cy_local = new_center

    cx_absolute = x_old + cx_local
    cy_absolute = y_old + cy_local

    new_x = cx_absolute - w // 2
    new_y = cy_absolute - h // 2
    return (new_x, new_y, w, h)

def crop_image(image, roi):
   x, y, w, h = roi
   return image[y:y+h, x:x+w] # .astype(np.float32) # TODO: esto del formato tal vez mejor función a parte.

def centered_roi(image, roi, show_changes=False):
    c = center(crop_image(image, roi)) # TODO: No es completamente necesario hacer esto en la imagen cropeada, se puede hacer en la original diría.
    new_roi = update_roi_center(roi, c)
    
    if show_changes:
        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')
        
        # Dibujar el old_roi
        x_old, y_old, w, h = roi
        rect_old = plt.Rectangle((x_old, y_old), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Old ROI')
        ax.add_patch(rect_old)
        
        # Dibujar el new_roi
        x_new, y_new, _, _ = new_roi
        rect_new = plt.Rectangle((x_new, y_new), w, h, linewidth=2, edgecolor='green', facecolor='none', label='New ROI')
        ax.add_patch(rect_new)
        
        # Mostrar la leyenda
        plt.legend(handles=[rect_old, rect_new], loc='upper right')
        plt.show()
    
    return new_roi

def center_and_crop(image, roi):
    new_roi = centered_roi(image, roi)
    cropped = crop_image(image, new_roi)
    return cropped

def correlate(template, image):
    correlation_map = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_map)

    return max_loc  

def track(template, image, blur=45):
    image_blur = cv2.GaussianBlur(image, (blur, blur), 100.90)
    template_blur = cv2.GaussianBlur(template, (blur, blur), 100.90)

    top_left = correlate(template_blur, image_blur)
    bottom_right = (top_left[0] + template.shape[0], top_left[1] + template.shape[1])
    new_center = [top_left[0] + template.shape[0]//2, top_left[1] + template.shape[1]//2] 

    image_matched = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return image_matched, new_center
