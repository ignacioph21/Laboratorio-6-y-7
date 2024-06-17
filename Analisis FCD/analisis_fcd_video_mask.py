import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal.windows import tukey
import matplotlib.animation as animation

from pathlib import Path
from skimage.io import imread, imsave

from pyfcd.fcd import calculate_carriers, fcd, normalize_image
from pyfcd.auxiliars import selectSquareROI, plot_fft, plot_height_field

# CONFIG 
i_ref_name = "0611-gota_con_jeringa.png"   # <-- CAMBIAR ACÁ (poner en Images/Referencias)
i_def_pattern = "gota_con_jeringa/202406_1457/*.bmp" # <-- CAMBIAR ACÁ (poner carpeta en Images/Displaced)
start = 2600

hp = 0.1              # [m]
alpha = 0.25
hstar = hp*alpha      # [m]

PXtoM = None          # [m]
square_size = 2.2e-3  # [m]

scale_roi_kwargs = None # {"width": 500, "height": 500}
roi = None

## CARGAR ARCHIVOS
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

images_path = dir_path.joinpath("Imagenes")
output_image_path = images_path.joinpath("Output")
reference_images_path = images_path.joinpath("Referencias")
displaced_images_path = images_path.joinpath("Displaced")
images = sorted([str(f) for f in displaced_images_path.glob(i_def_pattern)])

flag = cv2.IMREAD_UNCHANGED
i_ref = cv2.imread(str(reference_images_path.joinpath(i_ref_name)), flag)
i_def = cv2.imread(images[start], flag)

if roi is None:
    roi = selectSquareROI("i_def: seleccionar region de interes", i_def) # Orden del roi: (x,y,w,h).   
    cv2.destroyWindow("i_def: seleccionar region de interes")
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

x, y, w, h = roi
i_ref = np.array(i_ref, dtype=np.float32)[y:y+h, x:x+w]  
i_def = np.array(i_def, dtype=np.float32)[y:y+h, x:x+w]

window1dx = np.abs(tukey(roi[-1], 0.2))
window1dy = np.abs(tukey(roi[-2], 0.2))
window2d = np.sqrt(np.outer(window1dx, window1dy))

i_ref *= window2d
i_def *= window2d

carriers = calculate_carriers(i_ref, PXtoM, square_size=square_size, show_carriers=False)

fig = plt.figure( figsize=(8,8) )
im = plt.imshow(fcd(i_def, carriers, h=hstar, unwrap=True, show_angles=False))

def update(frame):
    i_def = cv2.imread(images[start+frame], flag)
    i_def = np.array(i_def, dtype=np.float32)[y:y+h, x:x+w]
    i_def *= window2d

    height_field = fcd(i_def, carriers, h=hstar, unwrap=True, show_angles=False) 
    im.set_array(height_field)

    return im

anim = animation.FuncAnimation(fig, 
                               update, 
                               frames = 1000,
                               interval = 1000 / 1000, # in ms
                               )
plt.show()
