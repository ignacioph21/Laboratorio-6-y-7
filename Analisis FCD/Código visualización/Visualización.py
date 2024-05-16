import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.fft import fft2, fftshift, fftfreq
from pathlib import Path
from skimage.io import imread, imsave

from pyfcd.fcd import calculate_carriers, fcd, normalize_image
from pyfcd.auxiliars import selectSquareROI, plot_fft, plot_height_field


roi = None
allowed_formats = "tiff, tif, bmp, png"  

reference_images_path = f".{os.sep}Imagenes{os.sep}Referencias"  
displaced_images_path = f".{os.sep}Imagenes{os.sep}Displaced"
theoreticals_image_path = f".{os.sep}Imagenes{os.sep}Teoricas"
name = "gota_f20_g70.png"

i_ref = cv2.imread(reference_images_path + os.sep + name, cv2.IMREAD_UNCHANGED)
i_def = cv2.imread(displaced_images_path + os.sep + name, cv2.IMREAD_UNCHANGED)
i_teo = cv2.imread(theoreticals_image_path + os.sep + name, cv2.IMREAD_UNCHANGED)

if roi is None:
    roi = selectSquareROI("i_def: seleccionar región de interés", i_def) # Orden del roi: (x,y,w,h).   
    cv2.destroyWindow("i_def: seleccionar región de interés")
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

i_ref = np.array(i_ref, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]  
i_def = np.array(i_def, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]  

plot_fft(i_ref, i_def)

print(f'processing reference image...', end='') # TODO: Cambiar el texto.
carriers = calculate_carriers(i_ref, show_carriers=True)
print('done')

t0 = time.time()
height_field = fcd(i_def, carriers, unwrap=True, show_angles=True) 
print(f'done in {time.time() - t0:.2}s\n')

plot_height_field(height_field, i_teo, roi)
