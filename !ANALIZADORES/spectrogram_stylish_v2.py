from matplotlib.widgets import TextBox, Button
from matplotlib.colors import LogNorm
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt
from tkinter import filedialog
import numpy as np
import datetime
import h5py

plt.rc('font', size=20)

# Configuración inicial
file_name = filedialog.askopenfilename(title="Seleccionar medición.")  
ranges = (0, 2000)  
fps = 125 # 500  
nperseg = 1028 
noverlap = 1017

pad = (nperseg//2)
crop = False # True  

# Leer datos desde el archivo HDF5
with h5py.File(file_name, 'r') as f:
    c = f['data'][0].shape[0] // 2
    centers = f['data'][ranges[0]:ranges[1], c, c] * 1e6

# Calcular tiempos
times = np.arange(len(centers)) / fps 

# Crear figura para la visualización
fig_main = plt.figure(figsize=(14, 6))  
gs = fig_main.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  # Añadir espacio para la barra de color
gs.update(wspace=0.4525)

# Subgráficos
ax3 = fig_main.add_subplot(gs[0, 0])    # Espectrograma en la parte inferior izquierda
ax2 = fig_main.add_subplot(gs[0, 1])    # Gráfico de evolución en la parte inferior central

# Eje dedicado para la barra de color
cbar_ax = fig_main.add_subplot(gs[0, 2])  # Barra de color al costado derecho del espectrograma

# Configurar ajuste inicial de la figura
fig_main.subplots_adjust(hspace=0.4, top=0.8936, bottom=0.1861476) 

# Función para actualizar el espectrograma
def update_spectrogram():
    global f, t_spec, Sxx, plotted_frequencies
    ax2.clear()
    window = hamming(nperseg)
    SFT = ShortTimeFFT(window, hop=nperseg - noverlap, fs=fps, scale_to='magnitude')

    # Calcular la STFT
    Sxx = SFT.stft((centers - np.mean(centers)))
    Sxx = 2 * np.abs(Sxx)
    f = SFT.f
    t_spec = SFT.t(len(centers)) # Punto de incio de la ventana. Agrega media al principio y final de los datos.
    
    spectrogram_plot = ax2.pcolormesh(t_spec, f, np.abs(Sxx), shading='gouraud', cmap='Greys_r', rasterized=True) # , norm=LogNorm(vmin=0.4) # v # 1e0 ### $ 4  # 'plasma' #  
    # print(Sxx.shape)
    ax2.set_ylim(0, 20.3)
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_title("Espectrograma")

    # Actualizar la barra de color
    cbar_ax.clear()  # Limpiar el eje de la barra de color
    fig_main.colorbar(spectrogram_plot, cax=cbar_ax, orientation='vertical', label="Amplitud [$\mu$m]")  # , ticklocation='left' # Barra de color vertical
    cbar_ax.tick_params(direction='in')
    # cbar_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    # Limpiar el subplot de frecuencias
    ax3.clear()
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Amplitud [$\mu$m]")
    # ax3.set_xlim(times[0], times[-1])
    # ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.grid(True)
    ax3.set_title("Evolución de Frecuencias")
    plotted_frequencies.clear()
    
# Subplot 3: Evolución de frecuencias
plotted_frequencies = {}  # Diccionario para rastrear frecuencias ya graficadas

# Inicializar el espectrograma
update_spectrogram()

# Función para graficar evolución de una frecuencia
def plot_frequency_evolution(frequency):
    if frequency in plotted_frequencies:
        return  # Evitar graficar la misma frecuencia dos veces

    # Buscar la evolución de amplitud en la frecuencia
    freq_idx = np.argmin(np.abs(f - frequency))
    amplitude_evolution = np.abs(Sxx[freq_idx, :])
    
    # Graficar en el subplot acumulativo
    t_window = nperseg // fps
    if crop == True:
        idx1 = np.argmin(abs(t_spec - t_window))
        idx2 = np.argmin(abs(t_spec - (t_spec[-1] - t_window)))
    elif crop == False:
        idx1 = 0  
        idx2 = -1  
    line, = ax3.plot(t_spec[idx1:idx2], amplitude_evolution[idx1:idx2], label=f"{frequency:.2f} Hz", marker=".")
    plotted_frequencies[frequency] = line
    ax3.legend(prop = {"size":10})

# Callback para procesar texto ingresado
text = input("Frecuencias (Hz): ")
frequencies = [float(f.strip()) for f in text.split(",")]
for freq in frequencies:
    plot_frequency_evolution(freq)
fig_main.canvas.draw_idle()


plt.savefig(f"spectrograms/{file_name.split('/')[-1][:-3]}.png", dpi=300, transparent = True)
plt.savefig(f"spectrograms/{file_name.split('/')[-1][:-3]}.pdf", dpi=300)




# Mostrar figura
plt.show()
