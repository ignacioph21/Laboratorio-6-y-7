from pyfcd.auxiliars import selectSquareROI
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from Tracking_blur import * 
import numpy as np
import cv2
import os

# CONFIG
i_ref_name = "Referencia.png"
i_tor_name = "Toroide.png"
i_def_pattern = f"E:{os.sep}Ignacio Hernando{os.sep}0611{os.sep}toroide_con_arena_forzado{os.sep}202406_1445"
start = 607

roi = None

images = os.listdir(i_def_pattern)

flag = cv2.IMREAD_UNCHANGED
i_ref = cv2.imread(i_ref_name, flag).astype(np.float32)
i_def = cv2.imread(i_def_pattern + os.sep + images[start], flag)
i_tor = cv2.imread(i_tor_name, flag)

if roi is None:
    roi = selectSquareROI("i_def: seleccionar region de interes", i_tor)  # Orden del roi: (x,y,w,h).
    cv2.destroyWindow("i_def: seleccionar region de interes")
    print("roi:", roi)  # por si queremos volver a seleccionar la misma región

""" Carriers """
i_ref = crop_image(i_ref, roi).astype(np.float32)

""" Toride de referencia """
template = center_and_crop(i_tor, roi)
toroid_mask = mask(template)

""" Trackeo """
i_def, c = track(template, i_def)
centers = [c]

fig = plt.figure( figsize=(8,8) )
im = plt.imshow(i_def) 


def update(frame):
    i_def_raw = cv2.imread(i_def_pattern + os.sep + images[start+frame], flag)
    i_def_raw, c = track(template, i_def_raw) # "! .astype(np.float32)
    centers.append(c)
    im.set_array(i_def_raw)

    return im, 

anim = animation.FuncAnimation(fig, 
                               update, 
                               frames = 1000,
                               interval = 1000 / 1000, # in ms
                               )

plt.show()

plt.scatter(*np.array(centers).T)
plt.show()
