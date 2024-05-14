import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pyfcd.fcd import calculate_carriers, fcd, normalize_image
from pathlib import Path
from skimage.io import imread, imsave

roi = None 
allowed_formats = "tiff, tif, bmp"

reference_images_path = ".\Referencia"
displaced_images_path = ".\Gota"

reference_images = os.listdir(reference_images_path)
displaced_images = os.listdir(displaced_images_path)


promediado = None 
max_iter = 1000 # 0
N = 0

for displaced in displaced_images:
    for reference in reference_images:
        if reference[-3:] in allowed_formats and  displaced[-3:] in allowed_formats:
            if N > max_iter:
                break
            N += 1

            i_ref = cv2.imread(reference_images_path + os.sep + reference, cv2.IMREAD_UNCHANGED)
            i_def = cv2.imread(displaced_images_path + os.sep + displaced, cv2.IMREAD_UNCHANGED)
            
                
            if roi is None:
                roi = cv2.selectROI("i_def: seleccionar región de interés", i_def) # Orden del roi: (x,y,w,h). 
                cv2.destroyWindow("i_def: seleccionar región de interés")
                print("roi:", roi) # por si queremos volver a seleccionar la misma región

            promediado = np.zeros((roi[-1], roi[-2]))
            i_ref = np.array(i_ref, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]
            i_def = np.array(i_def, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]

            # i_ref -= np.mean(i_ref)
            # i_def -= np.mean(i_def)

            # print(f'processing reference image...', end='') #TODO: Cambiar el texto.
            carriers = calculate_carriers(i_ref) #TODO: No entiendo porque le restamos el promedio a i_ref si despues no lo usamos
            # print('done')

            # t0 = time.time()
            height_field = fcd(i_def, carriers)
            # print(f'done in {time.time() - t0:.2}s\n')

            promediado += height_field
            # print(N) 

promediado /= N
plt.imshow(promediado)
plt.show()
