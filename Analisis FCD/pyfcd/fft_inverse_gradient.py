import numpy as np
from numpy.fft import fftfreq
from scipy.fft import fft2, ifft2

# Based on work from Sander Wildeman, which based it on the following paper:
#   Huhn, et al. Exp Fluids (2016), 57, 151, https://doi.org/10.1007/s00348-016-2236-3

def fftinvgrad(fx, fy, cal):
    size = fx.shape

    # the fourier method will implicitly subtract mean from gradient to satisfy
    # the periodicity assumption, we will tag it back on later
    mx = np.mean(fx)
    my = np.mean(fy)

    ky, kx = np.meshgrid(fftfreq(size[0], cal/(2*np.pi)), fftfreq(size[1], cal/(2*np.pi)), indexing='ij')

    # pre-compute k^2
    k2 = kx**2 + ky**2

    if size[1] % 2 == 0:
        kx[:,size[1]//2+1] = 0 # remove degeneracy at kx=Nx/2 leading to imaginary part

    if size[0] % 2 == 0:
        ky[size[0]//2+1,:] = 0 # remove degeneracy at ky=Ny/2 leading to imaginary part

    # compute fft of gradients
    fx_hat = fft2(fx)
    fy_hat = fft2(fy)

    # integrate in fourier domain
    k2[0,0] = 1 # shortcut to prevent division by zero (this effectively subtracts a linear plane)
    f_hat = (-1.0j * kx * fx_hat + -1.0j * ky * fy_hat) / k2

    # transform back to spatial domain
    f = np.real(ifft2(f_hat))

    #  add mean slope back on
    y, x = np.meshgrid(range(size[0]), range(size[1]), indexing='ij') # 
    # f = f + mx*x*cal**2 + my*y*cal**2 

    return f
