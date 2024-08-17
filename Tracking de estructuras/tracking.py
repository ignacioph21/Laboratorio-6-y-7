import matplotlib.pyplot as plt
import numpy as np
import cv2

def selectScaledROI(window_name, img, width=500, height=500):
    """
    Escala la imagen a las dimensiones especificadas, permite la selección de un ROI,
    y convierte las coordenadas del ROI escalado a la imagen original.

    :param window_name: Nombre de la ventana donde se mostrará la imagen.
    :param img: Imagen original donde se seleccionará el ROI.
    :param width: Ancho de la imagen escalada.
    :param height: Alto de la imagen escalada.
    :return: Coordenadas del ROI en la imagen original (x, y, w, h).
    """
    # Guarda las dimensiones originales de la imagen
    height_original, width_original = img.shape[:2]
    
    # Escala la imagen a las dimensiones especificadas
    img_scaled = cv2.resize(img, (width, height))
    
    # Utiliza selectROI para seleccionar un ROI rectangular en la imagen escalada
    rect_scaled = cv2.selectROI(window_name, img_scaled)
    
    # Extrae las coordenadas y dimensiones del ROI escalado
    x_scaled, y_scaled, w_scaled, h_scaled = rect_scaled
    
    # Calcula los factores de escala
    scale_x = width_original / width
    scale_y = height_original / height
    
    # Convierte las coordenadas del ROI escalado a la imagen original
    x_original = int(x_scaled * scale_x)
    y_original = int(y_scaled * scale_y)
    w_original = int(w_scaled * scale_x)
    h_original = int(h_scaled * scale_y)
    
    return (x_original, y_original, w_original, h_original)

def selectSquareROI(window_name, img, scale_kwargs={"width": 500, "height": 500}):
    # Utiliza selectROI para seleccionar un ROI rectangular
    if scale_kwargs is not None:
        rect = selectScaledROI(window_name, img, **scale_kwargs) # cv2.
    else:
        rect = cv2.selectROI(window_name, img) #TODO: Organizar esto un poco.

    # Extrae las coordenadas y dimensiones del ROI
    x, y, w, h = rect
    
    # Encuentra la dimensión más pequeña (ancho o alto)
    size = min(w, h)
    
    # Ajusta el ROI seleccionado para que sea un cuadrado
    x_adjusted = x + (w - size) // 2
    y_adjusted = y + (h - size) // 2
    square_roi = (x_adjusted, y_adjusted, size, size)
    
    return square_roi





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

def get_center(image, low=20, high=40, show_center=False): 
    image = image.astype(np.uint8)
    circles = cv2.HoughCircles(edges(image, low, high), cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=40, param2=20, minRadius=40, maxRadius=80)
    center = (int(circles[0][0, 0]), int(circles[0][0, 1])) if circles is not None else None
    if show_center and c:
        plt.imshow(image)
        plt.scatter([center[0]], [center[1]])
        plt.show()
    return center  

def local_to_absolute(local_coordinates, roi):
    return [local_coordinates[0]+roi[0], local_coordinates[1]+roi[1]]
    
def update_roi_center(old_roi, new_center):
    x_old, y_old, w, h = old_roi
    cx, cy = new_center

    new_x = cx - w // 2
    new_y = cy - h // 2
    return (new_x, new_y, w, h)

def roi_from_center(center, dimensions):
    return (center[0]-dimensions[0]//2, center[1]-dimensions[1]//2, dimensions[0], dimensions[1])

def roi_from_corner(corner, dimensions):
    return (corner[0], corner[1], dimensions[0], dimensions[1])

def get_roi_center(roi):
    return (roi[0] + roi[2]//2, roi[1] + roi[3]//2)

def crop_image(image, roi):
   x, y, w, h = roi
   return image[y:y+h, x:x+w]

def rotate_image(image, angle, center = None):
    if center is None:
        center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def center_and_crop(image, roi, rotate = 0):
    c_local = get_center(crop_image(image, roi)) # TODO: No es completamente necesario hacer esto en la imagen cropeada, se puede hacer en la original diría.
    c = local_to_absolute(c_local, roi)
    if rotate:
        image = rotate_image(image, rotate, c)
    new_roi = update_roi_center(roi, c)
    cropped = crop_image(image, new_roi)
    return cropped

def correlate(template, image):
    correlation_map = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_map)

    # plt.imshow(correlation_map)
    # plt.show()
    
    return max_loc  

def expand_roi(roi, factor=5):
    return (roi[0]-factor, roi[1]-factor, roi[2]+2*factor, roi[3]+2*factor)

def add_gaussian_noise(image, mean=0, std=5):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image
 
def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    noisy_image = image.copy()
    h, w = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
 
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = 0 
        else:
            noisy_image[row, col] = 255
 
    return noisy_image

def track(template, image, blur=0):
    if blur:
        image_to_correlate = cv2.GaussianBlur(image, (blur, blur), 100.90)
        template_to_correlate = cv2.GaussianBlur(template, (blur, blur), 100.90)
    else:
        image_to_correlate = image
        template_to_correlate = template
        
    top_left = correlate(template_to_correlate, image_to_correlate)
    bottom_right = (top_left[0] + template.shape[0], top_left[1] + template.shape[1])
    new_center = [top_left[0] + template.shape[0]//2, top_left[1] + template.shape[1]//2] 

    image_matched = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return image_matched, new_center

def track_optimized(template, image, search_roi=None, blur=0, factor=5):    
    if blur:
        image_to_correlate = cv2.GaussianBlur(image, (blur, blur), 100.90)
        template_to_correlate = cv2.GaussianBlur(template, (blur, blur), 100.90)
    else:
        image_to_correlate = image
        template_to_correlate = template

    if search_roi is None:
        search_roi = roi_from_corner((0, 0), image_to_correlate.shape)
    else:
        search_roi = expand_roi(search_roi, factor)
    image_to_correlate = crop_image(image_to_correlate, search_roi)
        
    top_left_local = correlate(template_to_correlate, image_to_correlate)
    top_left = local_to_absolute(top_left_local, search_roi) 
    new_roi = roi_from_corner(top_left, template.shape)    
    new_center = get_roi_center(new_roi)
    
    image_matched = crop_image(image, new_roi)

    return image_matched, new_roi, new_center


def track_mean(template, image, blur=0):
    if blur:
        image_to_correlate = cv2.GaussianBlur(image, (blur, blur), 10)
        template_to_correlate = cv2.GaussianBlur(template, (blur, blur), 10)
    else:
        image_to_correlate = image
        template_to_correlate = template
        
    top_left1 = np.array(correlate(template_to_correlate[:,::-1],    image_to_correlate))
    top_left2 = np.array(correlate(template_to_correlate[::-1,:],    image_to_correlate))
    top_left3 = np.array(correlate(template_to_correlate[:,:],       image_to_correlate))
    top_left4 = np.array(correlate(template_to_correlate[::-1,::-1], image_to_correlate))
    top_left = (top_left1 + top_left2 + top_left3 + top_left4)//4
    
    bottom_right = (top_left[0] + template.shape[0], top_left[1] + template.shape[1])
    new_center = [top_left[0] + template.shape[0]//2, top_left[1] + template.shape[1]//2] 

    image_matched = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return image_matched, new_center
