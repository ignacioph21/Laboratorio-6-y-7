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

roi = None
allowed_formats = "tiff, tif, bmp, png"  

name = "gota6.png"   # <-- CAMBIAR ACÁ #TODO: dejo para que pruebes y compares los resultados con los que tenés ahora

hp = 0.026          # [m]
alpha = 0.25
hstar = hp*alpha    # [m]

PXtoM = 1e-3/13.7   # [m]
square_size = None  # [m]

scale_roi_kwargs = {"width": 500, "height": 500}
roi = None

## CARGAR ARCHIVOS
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
images_path = dir_path.joinpath("Imagenes")

reference_images_path = images_path.joinpath("Referencias")
displaced_images_path = images_path.joinpath("Displaced")
theoreticals_image_path = images_path.joinpath("Teoricas")
output_image_path = images_path.joinpath("Output")

flag = cv2.IMREAD_UNCHANGED
i_ref = cv2.imread(str(reference_images_path.joinpath(name)), flag)
i_def = cv2.imread(str(displaced_images_path.joinpath(name)), flag)
i_teo = cv2.imread(str(theoreticals_image_path.joinpath(name)), flag)

if roi is None:
    roi = selectSquareROI("i_def: seleccionar region de interes", i_def) # Orden del roi: (x,y,w,h).   
    cv2.destroyWindow("i_def: seleccionar region de interes")
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

x, y, w, h = roi
i_ref = np.array(i_ref, dtype=np.float32)[y:y+h, x:x+w]  
i_def = np.array(i_def, dtype=np.float32)[y:y+h, x:x+w]

window1dx = np.abs(tukey(roi[-1], 0.1))
window1dy = np.abs(tukey(roi[-2], 0.1))
window2d = np.sqrt(np.outer(window1dx, window1dy))

i_ref *= window2d
i_def *= window2d

plot_fft(i_ref, i_def)

print(f'Calculando carriers...', end='\n') # TODO: no es necesario siempre calcular los carriers.
carriers = calculate_carriers(i_ref, PXtoM, square_size=square_size, show_carriers=False)
print('Carriers calculados. Iniciando procesado de imagen deformada.')
t0 = time.time()
height_field = fcd(i_def, carriers, h=hstar, unwrap=True, show_angles=False) 
print(f'Finalizado en {time.time() - t0:.2} s.\n')

plot_height_field(height_field, i_teo=i_teo, roi=roi, PXtoM=carriers[0].PXtoM)
