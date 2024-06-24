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
from pyfcd.image_process import *

# CONFIG 
i_ref_name = "Referencia.png" 
i_def_pattern = f"E:{os.sep}Ignacio Hernando{os.sep}0611{os.sep}toroide_con_arena_forzado{os.sep}202406_1445" # 202406_1247  # <-- CAMBIAR ACÁ (poner carpeta en Images/Displaced)
start = 607

hp = 0.1              # [m]
alpha = 0.25
hstar = hp*alpha      # [m]

PXtoM = None          # [m]
square_size = 2.2e-3  # [m]

scale_roi_kwargs = None # {"width": 500, "height": 500}
roi = None

images = os.listdir(i_def_pattern)

flag = cv2.IMREAD_UNCHANGED
i_ref = cv2.imread(i_ref_name, flag)
i_def_raw = cv2.imread(i_def_pattern + os.sep + images[start], flag)

if roi is None:
    roi = selectSquareROI("i_def: seleccionar region de interes", i_def_raw) # Orden del roi: (x,y,w,h).   
    cv2.destroyWindow("i_def: seleccionar region de interes")
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

x, y, w, h = roi
i_ref = np.array(i_ref, dtype=np.float32)[y:y+h, x:x+w]  
i_def = np.array(i_def_raw, dtype=np.float32)[y:y+h, x:x+w]

carriers = calculate_carriers(i_ref, PXtoM, square_size=square_size, show_carriers=False)
i_def = windowed(masked(i_def, i_ref, N=15), 0.2) # , low=140 

fig = plt.figure( figsize=(8,8) )
plt.subplot(121)
im_original = plt.imshow(i_def)
plt.subplot(122)
im = plt.imshow(fcd(i_def, carriers, h=hstar, unwrap=True, show_angles=False)) # TODO: hay que definir bien los máximons y mínimos para que no sature. 


def roi_from_center(cx, cy, w, h):
    return (cx-w//2, cy-h//2, w, h)

def update_roi(image, old_roi):
    x, y, w, h = old_roi
    new_center = center(image[y:y+h, x:x+w], 20, 40) # 100 200 
    if new_center:
        cx_, cy_ = new_center
        new_roi = roi_from_center(cx_+x, cy_+y, w, h)
        return new_roi
    else:
        print("Now.")
        return old_roi

centers = []    
l = 10

def update(frame):
    i_def_raw = cv2.imread(i_def_pattern + os.sep + images[start+frame], flag)
    x, y, w, h = update_roi(i_def_raw, roi)
    i_def = np.array(i_def_raw, dtype=np.float32)[y:y+h, x:x+w]     
    i_def = windowed(masked(i_def, i_ref, N=15, low=80), 0.2) # , low=140 

    height_field = fcd(i_def, carriers, h=hstar, unwrap=True, show_angles=False) 
    centers.append(np.mean(height_field[w//2-l:w//2+l,h//2-l:h//2+l]))
    im.set_array(height_field)
    im_original.set_array(i_def)

    return im_original, im, 

anim = animation.FuncAnimation(fig, 
                               update, 
                               frames = 1000,
                               interval = 1000 / 1000, # in ms
                               )
plt.show()

ts = np.linspace(0, len(centers), len(centers))/250*1000
plt.scatter(ts, np.array(centers)*1000) # a 
plt.show()    
