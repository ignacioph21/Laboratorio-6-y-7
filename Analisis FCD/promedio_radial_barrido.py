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


n_aire = 1          # Índice de refracción del aire
n_agua = 1.33       # Índice de refracción del agua
n_vidrio = 1.5      # Índice de refracción del vidrio
alpha = 1-n_aire/n_agua
    
PXtoM = None       # [m] #TODO: Lo que era el factor de calibración cal
square_size = 1e-3 # [m]

scale_roi_kwargs = {"width": 500, "height": 500}
roi = None # (348, 362, 249, 249) # None # (255, 287, 399, 399)

i_teo = None

## CARGAR ARCHIVOS
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
images_path = dir_path.joinpath("Imagenes")

reference_images_path = images_path.joinpath("Referencias")
displaced_images_path = images_path.joinpath("Displaced")
theoreticals_image_path = images_path.joinpath("Teoricas")
output_image_path = images_path.joinpath("Output")


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return np.linspace(np.min(r), np.max(r), len(radialprofile)), radialprofile


for i in range(1, 8):
    name = f"0531-Barrido_una_gota_{i}mm.png"     

    flag = cv2.IMREAD_UNCHANGED
    i_ref = cv2.imread(str(reference_images_path.joinpath(name)), flag)
    i_def = cv2.imread(str(displaced_images_path.joinpath(name)), flag)
    i_teo = cv2.imread(str(theoreticals_image_path.joinpath(name)), flag)

    ## ROI: CROPPEAR IMAGEN
    if roi is None:
        # roi = selectSquareROI("i_def: seleccionar region de interes", i_def, scale_kwargs=scale_roi_kwargs)
        roi = cv2.selectROI("i_def: seleccionar region de interes", i_def)
        cv2.destroyWindow("i_def: seleccionar region de interes")
        print("roi:", roi) # por si queremos volver a seleccionar la misma región

    x, y, w, h = roi
    i_ref = np.array(i_ref, dtype=np.float32)[y:y+h, x:x+w]  
    i_def = np.array(i_def, dtype=np.float32)[y:y+h, x:x+w]

    ## VENTANA PARA MEJORAR FFT'S
    window1dx = np.abs(tukey(roi[-1], 0.005))
    window1dy = np.abs(tukey(roi[-2], 0.005))
    window2d = np.sqrt(np.outer(window1dx, window1dy))

    i_ref *= window2d
    i_def *= window2d

    hp = (i-1)*1e-3*n_agua/n_aire + 1e-3*n_agua/n_vidrio
    hstar = hp*alpha   # [m]
    print(f"Altura efectiva aire: {(i-1)*1e-3*n_agua/n_aire*1000} mm.")


    ## ANALISIS FCD
    print(f'Calculando carriers...', end='\n') # TODO: no es necesario siempre calcular los carriers.
    carriers = calculate_carriers(i_ref, PXtoM, square_size=square_size, show_carriers=False)
    print('Carriers calculados. Iniciando procesado de imagen deformada.')
    t0 = time.time()
    height_field = fcd(i_def, carriers, h=hstar, unwrap=True, show_angles=False) 
    print(f'Finalizado en {time.time() - t0:.2} s.\n')

    maximum = np.argwhere(height_field==np.max(height_field))[0]
    r, promedio = radial_profile(height_field, maximum[::-1])
    plt.plot(r*carriers[0].PXtoM*1000, promedio*1000, label=f"Altura efectiva: {hp*1000:.2f} mm.")

plt.legend()
plt.xlabel("Radio [mm]")
plt.ylabel("Altura [mm]")

plt.show()
