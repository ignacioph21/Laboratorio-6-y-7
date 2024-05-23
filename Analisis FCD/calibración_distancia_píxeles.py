import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal.windows import tukey
from scipy.signal import find_peaks

from pyfcd.auxiliars import selectScaledROI
from pathlib import Path
from skimage.io import imread, imsave



roi = None
allowed_formats = "tiff, tif, bmp, png"  

image_path = f".{os.sep}Imagenes{os.sep}Displaced"
name = "gota6_v2.png"

image = cv2.imread(image_path + os.sep + name, cv2.IMREAD_UNCHANGED)

if roi is None:
    roi = selectScaledROI("Seleccionar regla", image) # Orden del roi: (x,y,w,h).   
    cv2.destroyWindow("Seleccionar regla") 
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

image = np.array(image, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]


promedio = 0
for row in image:
    peaks = find_peaks(-row, distance=7.5, prominence=0.5)[0]
    promedio += np.mean(np.diff(peaks))
    # plt.plot(row) # peaks
    # plt.scatter(peaks, row[peaks])
    # plt.show()

promedio /= len(image)
print(f"Calibración: {promedio:.3f} px/mm." )    
plt.show()
