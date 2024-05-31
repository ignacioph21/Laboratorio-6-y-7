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

def generate_checkerboard_from_mesh(y, x, interval):
    return 0.5 + (np.cos(x * np.pi / interval[0]) + np.cos(y * np.pi / interval[1])) / 4.0

# Cargar los datos desde el archivo CSV
data = np.loadtxt(f'Generate test images{os.sep}output50.csv', delimiter=',', skiprows=1)  

# Parámetros fijos.
scale = 3
resolution = 256     
size = (resolution, resolution)

square_size = 0.51e-3
pattern_interval =  np.array([square_size, square_size]) 

# Extraer los valores de x e y
r = np.array(list(-data[:, 1][::-1]) + list(data[:, 1]))
h = np.array(list(-data[:, 2][::-1]) + list(-data[:, 2]))

cal = (2*scale*max(r))/resolution
print(f" Calibración: {cal*1000:.3f} mm/px.")

xs = np.linspace(-1/2, 1/2, resolution)*(2*scale*max(r)) 
ys = np.linspace(-1/2, 1/2, resolution)*(2*scale*max(r)) 
y, x = np.meshgrid(xs, ys, indexing='ij')

rs = np.sqrt(x**2+y**2)
height_field = np.interp(rs, r, h)

height_field_gradient_y, height_field_gradient_x  = np.gradient(height_field, xs, ys)

# Inicia procesado para múltiples H.
Hs = np.linspace(0, 30, 100)/1000      

window1d = np.abs(tukey(resolution, 0.1))
window2d = np.sqrt(np.outer(window1d,window1d))

i_ref = generate_checkerboard_from_mesh(x, y, pattern_interval)  
i_ref *= window2d
carriers = calculate_carriers(i_ref, cal, show_carriers=True)

coefficients = []
for H in Hs:
    gain = -H*0.25  
    i_def = generate_checkerboard_from_mesh(y + gain*height_field_gradient_y,
                                            x + gain*height_field_gradient_x,
                                            pattern_interval)  
    i_def *= window2d
    height_field_processed = fcd(i_def, carriers, cal, unwrap=True)     
    height_field_processed /= -H*0.25
    coefficients.append(np.max(height_field)/np.max(height_field_processed))
    if H>0.014:
        pass
        # plot_height_field(height_field_processed, None, None)  
    print(H)

plt.plot(Hs, coefficients)
plt.show()      

