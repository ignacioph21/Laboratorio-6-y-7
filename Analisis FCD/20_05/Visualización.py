import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal.windows import tukey

from pathlib import Path
from skimage.io import imread, imsave

from pyfcd.fcd import calculate_carriers, fcd, normalize_image
from pyfcd.auxiliars import selectSquareROI, plot_fft, plot_height_field

cal = 1.4582923107942913e-05 # 1e-3/13.705308493589744    

roi = None
allowed_formats = "tiff, tif, bmp, png"  

reference_images_path = f".{os.sep}Imagenes{os.sep}Referencias"  
displaced_images_path = f".{os.sep}Imagenes{os.sep}Displaced"
theoreticals_image_path = f".{os.sep}Imagenes{os.sep}Teoricas"
name = "gota_super4.png"   

i_ref = cv2.imread(reference_images_path + os.sep + name, cv2.IMREAD_UNCHANGED)
i_def = cv2.imread(displaced_images_path + os.sep + name, cv2.IMREAD_UNCHANGED)
i_teo = cv2.imread(theoreticals_image_path + os.sep + name, cv2.IMREAD_UNCHANGED)

if roi is None:
    roi = selectSquareROI("i_def: seleccionar región de interés", i_def) # Orden del roi: (x,y,w,h).   
    cv2.destroyWindow("i_def: seleccionar región de interés")
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

i_ref = np.array(i_ref, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]  
i_def = np.array(i_def, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]  

window1d = np.abs(tukey(roi[-1], 0.1))
window2d = np.sqrt(np.outer(window1d,window1d))

i_ref *= window2d
i_def *= window2d

plot_fft(i_ref, i_def)

print(f'processing reference image...', end='') # TODO: Cambiar el texto.
carriers = calculate_carriers(i_ref, cal, show_carriers=True)
print('done')

t0 = time.time()
height_field = fcd(i_def, carriers, cal, unwrap=True, show_angles=True) 
print(f'done in {time.time() - t0:.2}s\n')


alpha = -0.026*0.25
print(np.max(height_field/alpha)*1000)
plot_height_field(height_field/alpha, i_teo, roi, cal)
