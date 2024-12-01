from skimage.transform import warp_polar
import matplotlib.pyplot as plt
import numpy as np

amplitudes = np.load('amplitude_5.25Hz_Exponencial_grande_2_202411_1112.h5.npy')

def radial_profile(data, center=None, max_radius=None):
    if center is None:
        c = data.shape[0]//2
        center = (c, c)
    if max_radius is None:
        max_radius = -1
    
    polars = warp_polar(data, center)
    radial_mean = np.mean(polars, axis=0)[:max_radius]
    radial_error = np.std(polars, axis=0)[:max_radius]/np.sqrt(max_radius)
    
    return np.linspace(0, max_radius, max_radius), radial_mean, radial_error



rs, r, err_r = radial_profile(amplitudes, max_radius=200)
plt.errorbar(rs, r, err_r, ls="none", capsize=2)
plt.vlines(180, min(r), max(r), color="black", linestyle="--")
plt.show()
