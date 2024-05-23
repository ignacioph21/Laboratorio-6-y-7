# himport argparse
from itertools import product

import os
import imageio
import numpy as np
from numpy import array, gradient
from skimage.draw import disk, rectangle
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from fft_inverse_gradient import fftinvgrad

def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())


def shade(i, f, relative=False, dimensions=2):
    scale = 1.0 / np.array(i.shape[:dimensions]) if relative else 1.0
    for p in product(*map(range, i.shape[:dimensions])):
        i[p] = f(array(p) * scale)


def shade_new(shape, f, dtype=np.float64, relative=False, dimensions=2):
    i = np.zeros(shape, dtype=dtype)
    shade(i, f, relative, dimensions)
    return i


def displacement_magnitude(displacement_field):
    return np.sqrt(displacement_field[..., 0] ** 2 + displacement_field[..., 1] ** 2)


def generate_checkerboard_from_mesh(y, x, interval):
    return 0.5 + (np.cos(x * np.pi / interval[0]) + np.cos(y * np.pi / interval[1])) / 4.0


def generate_height_field_ripples(size):
    def height_at_point(p, wave_center, wave_interval, wave_amplitude=0.01, wave_falloff=10):
        r = np.sqrt(sum((p - wave_center) ** 2))
        height = wave_amplitude * np.cos(r * np.pi / wave_interval) * (0.05 / ((r + 0.5) ** wave_falloff + 0.2))
        return height

    return shade_new(size, lambda p: height_at_point(p,
                                                     wave_center=array([0.5, 0.5]),
                                                     wave_interval=0.1,
                                                     wave_amplitude=0.1),
                     relative=True)


def generate_height_field_smiley(size):
    result = np.zeros(size, dtype=float)
    result[disk((size[0] / 2, size[1] / 2), size[0] / 3, shape=size)] = 0.04
    result[disk((size[0] / 2, size[1] / 2), size[0] / 4, shape=size)] = 0
    result[tuple(rectangle(start=(0,0), end=(size[0]//2, size[1]), shape=size))] = 0

    result[disk((size[0] / 5, size[1] / 3), size[0] / 10, shape=size)] = 0.07
    result[disk((size[0] / 5, 2 * size[1] / 3), size[0] / 10, shape=size)] = 0.07


    result = gaussian(result, 10)
    return result


# Cargar los datos desde el archivo CSV
y = np.loadtxt('output50.csv', delimiter=',', skiprows=1)  

# Extraer los valores de x e y
scale = 3
square_size = 2e-3

gain = -0.026*0.25 

resolution = 256  * 2   
size = (resolution, resolution)

r = np.array(list(-y[:, 1][::-1]) + list(y[:, 1]))
h = np.array(list(-y[:, 2][::-1]) + list(-y[:, 2]))

print(f" Calibración: {(2*scale*max(r))/resolution*1000:.3f} mm/px.)
xs = np.linspace(-1/2, 1/2, resolution)*(2*scale*max(r)) 
ys = np.linspace(-1/2, 1/2, resolution)*(2*scale*max(r)) 
y, x = np.meshgrid(xs, ys, indexing='ij')

rs = np.sqrt(x**2+y**2)
height_field = np.interp(rs, r, h)

# height_field = generate_height_field_ripples(size)
height_field_gradient_y, height_field_gradient_x  = gradient(height_field, xs, ys)
# height_field_gradient_y, height_field_gradient_x  = gradient(generate_height_field_smiley(size))

##h = fftinvgrad(height_field_gradient_x, height_field_gradient_y, 1*(2*scale*max(r))/resolution)
##plt.imshow(h)
##plt.show() 


pattern_interval =  array([square_size, square_size]) 
displaced_checkerboard = generate_checkerboard_from_mesh(y + gain*height_field_gradient_y,
                                                         x + gain*height_field_gradient_x,
                                                         pattern_interval)

reference_checkerboard = generate_checkerboard_from_mesh(x, y, pattern_interval)

plt.imshow(displaced_checkerboard)
plt.axis("off")
plt.show()

name = input("Nombre del archivo: ")
imageio.imwrite(f'..{os.sep}Imagenes{os.sep}Referencias{os.sep}{name}.png', (reference_checkerboard * 255).astype(np.uint8))
imageio.imwrite(f'..{os.sep}Imagenes{os.sep}Displaced{os.sep}{name}.png', (displaced_checkerboard * 255).astype(np.uint8))
imageio.imwrite(f'..{os.sep}Imagenes{os.sep}Teoricas{os.sep}{name}.png', (normalize_image(height_field) * 255).astype(np.uint8))

