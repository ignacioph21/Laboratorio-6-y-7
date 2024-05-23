import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pyfcd.auxiliars import selectSquareROI, plot_height_field
from pyfcd.fcd import calculate_carriers, fcd, normalize_image
from pathlib import Path
from skimage.io import imread, imsave

roi = None
allowed_formats = "tiff, tif, bmp, png"  

reference_images_path = "Referencias7"
displaced_images_path = "Gotas7"

reference_images = os.listdir(reference_images_path)
displaced_images = os.listdir(displaced_images_path)


promediado_ref = np.zeros((1024, 1024)) 
promediado_def = np.zeros((1024, 1024)) 
max_iter = 100
N = 0


for (reference, displaced) in zip(reference_images, displaced_images): 
    if reference[-3:] in allowed_formats and displaced[-3:] in allowed_formats:
        if N > max_iter:
            break
        N += 1

        i_ref = cv2.imread(reference_images_path + os.sep + reference, cv2.IMREAD_UNCHANGED)
        i_def = cv2.imread(displaced_images_path + os.sep + displaced, cv2.IMREAD_UNCHANGED)
        
        i_ref = np.array(i_ref, dtype=np.float32)
        i_def = np.array(i_def, dtype=np.float32)

        promediado_ref += i_ref
        promediado_def += i_def

promediado_ref /= N
promediado_def /= N

plt.subplot(121)
plt.imshow(promediado_ref)
plt.axis("off")

plt.subplot(122)
plt.imshow(promediado_def)
plt.axis("off")

plt.show()

name = input("Nombre de las fotos: ")
imsave(f"Imagenes{os.sep}Referencias{os.sep}{name}.png", (promediado_ref).astype(np.uint8))
imsave(f"Imagenes{os.sep}Displaced{os.sep}{name}.png", (promediado_def).astype(np.uint8))

plt.show()
