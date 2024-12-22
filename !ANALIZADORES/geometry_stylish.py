import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter import filedialog
from scipy.fft import rfft, rfftfreq
import h5py


plt.rc('font', size=22)

def pixel_to_mm(coord, cal=1):
    return coord * cal

def video_fft(frames, fps=1):
    frames_fft = rfft(frames, axis=0)
    frames_fft /= len(frames)
    freqs = rfftfreq(len(frames), 1 / fps)
    return freqs, frames_fft

def amplitude(freqs, frames_fft, freq_loc):
    differences = np.abs(freqs - freq_loc)
    index = np.argmin(differences)
    print(f"Diferencia con la frecuencia deseada: {differences[index]:.2e} Hz")
    return np.abs(frames_fft[index]), np.angle(frames_fft[index])

# Configuración inicial
fps = 125
cal = 0.00022318840579710144 * 1e3  # Conversión de px a mm

# Seleccionar frecuencia de interés
file_name = filedialog.askopenfilename(title="Seleccionar medición.") # "F:/Ignacio Hernando/Mediciones Procesadas/Toroide Grande/Exponencial_grande_2_202411_1112.h5"
freq_loc = float(input("Frecuencia a analizar (Hz): ")) # 10.6 # 5  # Ejemplo: frecuencia en Hz
r = int(input(f"Borde de la estructura (px): ")) # 200
r_mask = int(input(f"Borde de la máscara (px): "))
ranges = (200, 700)




# Leer datos desde el archivo HDF5
with h5py.File(file_name, 'r') as f:
    c = f['data'][0].shape[0] // 2
    frames = f['data'][ranges[0]:ranges[1]]
    frames = frames[:, c-r:c+r, c-r:c+r] * 1e6

# Calcular FFT del video
freqs, vidfft = video_fft(frames, fps=fps)

amplitudes, fases = amplitude(freqs, vidfft, freq_loc)
# amplitudes /= len(frames)

# Coordenadas y centro en píxeles
px_x = np.arange(frames.shape[2])
px_y = np.arange(frames.shape[1])
x_mm = pixel_to_mm(px_x, cal)
y_mm = pixel_to_mm(px_y, cal)
center_mm = pixel_to_mm(r, cal)  # Centro en mm
radius_mm = pixel_to_mm(r, cal)  # Radio en mm
radius_mask_mm = pixel_to_mm(r_mask, cal)  # Radio en mm

def set_custom_ticks(axis, data):
    ticks = np.array([data[0], data[len(data) // 2], data[-1]], dtype=int)
    axis.set_ticks(ticks)

def set_custom_colorbar_ticks(cbar, data):
    min_val = np.min(data)
    max_val = np.max(data)
    median_val = ( min_val + max_val ) / 2
    ticks = np.array([min_val, median_val, max_val], dtype=int)
    cbar.set_ticks(ticks)
    
def beautify_axis(ax, x_data, y_data, cbar=None, cbar_data=None):
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.tick_params(direction='in')
    set_custom_ticks(ax.xaxis, x_data)
    set_custom_ticks(ax.yaxis, y_data)
    if cbar is not None and cbar_data is not None:
        set_custom_colorbar_ticks(cbar, cbar_data)

# Crear figura
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico de amplitudes
im1 = axs[0].imshow(amplitudes, extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]], cmap="Greys_r")
axs[0].add_patch(Circle((center_mm, center_mm), radius_mm, color='red', fill=False, linewidth=4))
axs[0].add_patch(Circle((center_mm, center_mm), radius_mask_mm, color='red', linestyle="--", fill=False, linewidth=4))

divider = make_axes_locatable(axs[0])
cax = divider.new_vertical(size='5%', pad=0.5)
fig.add_axes(cax)
cbar1 = fig.colorbar(im1, cax=cax, label="Amplitud [$\mu$m]", orientation='horizontal', location="top")
cbar1.ax.tick_params(direction='in')

# Gráfico de fases
im2 = axs[1].imshow(fases, extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]], cmap="hsv", vmin=-np.pi, vmax=np.pi)
axs[1].add_patch(Circle((center_mm, center_mm), radius_mm, color='black', fill=False, linewidth=4))
axs[1].add_patch(Circle((center_mm, center_mm), radius_mask_mm, color='black', linestyle="--", fill=False, linewidth=4))

divider = make_axes_locatable(axs[1])
cax = divider.new_vertical(size='5%', pad=0.5)
fig.add_axes(cax)
cbar2 = fig.colorbar(im2, cax=cax, label="Fase [rad]", orientation='horizontal', location="top")
cbar2.ax.tick_params(direction='in')

# Aplicar formato
beautify_axis(axs[0], x_mm, y_mm, cbar1, amplitudes)
beautify_axis(axs[1], x_mm, y_mm, cbar2, fases)

# Ajustar diseño y guardar la figura
plt.tight_layout()
fig.subplots_adjust(left=0.054, bottom=0.13, right=0.976, top=0.895, wspace=0.114, hspace=0.2)
plt.savefig(f"outputs/amplitude_phase_{freq_loc}Hz.png", dpi=300, transparent = True)
plt.savefig(f"outputs/amplitude_phase_{freq_loc}Hz.pdf")
plt.show()
