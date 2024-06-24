from scipy.signal.windows import tukey
import numpy as np
import cv2


def windowed(image, x_factor, y_factor=None):
    y_factor = y_factor if y_factor else x_factor
    window1dx = np.abs(tukey(image.shape[0], x_factor))
    window1dy = np.abs(tukey(image.shape[1], y_factor))
    window2d = np.sqrt(np.outer(window1dx, window1dy))

    return image*window2d

def masked(i_def, i_ref, N=15, H=10, low=60, high=255):
    blurred_image = cv2.GaussianBlur(i_def, (N, N), H)

    _, binary = cv2.threshold(blurred_image, low, high, cv2.THRESH_BINARY)
    kernel = np.ones((N, N), np.uint8)
    mask_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)//255

    return i_def*mask_open + i_ref * (1-mask_open)

def mask(image, N=15, H=10, low=60, high=255):
    blurred_image = cv2.GaussianBlur(image, (N, N), H)    

    _, binary = cv2.threshold(blurred_image, low, high, cv2.THRESH_BINARY)
    kernel = np.ones((N, N), np.uint8)
    mask_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)//255

    return mask_open  

def edges(image, low, high):
    return cv2.Canny(mask(image)*255, low, high) 
    
def center(image, low, high):
    image = image.astype(np.uint8)
    circles = cv2.HoughCircles(edges(image, low, high), cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=40, param2=20, minRadius=40, maxRadius=80)
##    # Asegurarse de que se hayan detectado círculos
##    if circles is not None:
##        # Convertir las coordenadas (x, y) y el radio a enteros
##        circles = np.uint16(np.around(circles))
##
##        # Dibujar los círculos
##        for i in circles[0, :]:
##            # Dibujar el círculo en la imagen original
##            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
##            # Dibujar el centro del círculo
##            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
##
##    # Mostrar la imagen con los círculos detectados
##    cv2.imshow('Círculos detectados', image)
##    # cv2.waitKey(0)
##    cv2.destroyAllWindows()



    return (int(circles[0, 0][0]), int(circles[0, 0][1])) if circles is not None else None 
