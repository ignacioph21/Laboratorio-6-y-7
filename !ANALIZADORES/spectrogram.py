from matplotlib.widgets import TextBox, Button
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from tkinter import filedialog
import numpy as np
import datetime
import h5py

plt.rc('font', size=14)

# Configuración inicial
file_name = filedialog.askopenfilename(title="Seleccionar medición.")  
ranges = (0, 2050)
fps = 125  
nperseg = 1028 
noverlap = 1000  

# Leer datos desde el archivo HDF5
with h5py.File(file_name, 'r') as f:
    c = f['data'][0].shape[0] // 2
    centers = f['data'][ranges[0]:ranges[1], c, c]

centers = np.concatenate([[centers[0]]*1000, centers])

# Calcular tiempos
times = np.arange(len(centers)) / fps 

# Crear figura para la visualización
fig_main = plt.figure(figsize=(16, 9))
gs = fig_main.add_gridspec(2, 3, width_ratios=[1, 0.05, 1], height_ratios=[1, 2])  # Añadir espacio para la barra de color
gs.update(wspace=0.3525)

# Subgráficos
ax1 = fig_main.add_subplot(gs[0, :])  # Serie temporal ocupa la fila superior (sin incluir la columna de la barra de color)
ax2 = fig_main.add_subplot(gs[1, 0])    # Espectrograma en la parte inferior izquierda
ax3 = fig_main.add_subplot(gs[1, 2])    # Gráfico de evolución en la parte inferior central

# Eje dedicado para la barra de color
cbar_ax = fig_main.add_subplot(gs[1, 1])  # Barra de color al costado derecho del espectrograma

# Configurar ajuste inicial de la figura
fig_main.subplots_adjust(hspace=0.4, top=0.936, bottom=0.176) 

# Función para actualizar el espectrograma
def update_spectrogram():
    global f, t_spec, Sxx, plotted_frequencies
    ax2.clear()
    f, t_spec, Sxx = spectrogram((centers - np.mean(centers)) * 1e3, fps, window='hamming', nperseg=nperseg, noverlap=noverlap)
    spectrogram_plot = ax2.pcolormesh(t_spec, f, np.abs(Sxx), shading='gouraud', cmap='Grays', rasterized=True, norm=LogNorm()) # 'plasma' 
    # print(Sxx.shape)
    ax2.set_ylim(0, 20.3)
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_title("Espectrograma")

    # Actualizar la barra de color
    cbar_ax.clear()  # Limpiar el eje de la barra de color
    fig_main.colorbar(spectrogram_plot, cax=cbar_ax, orientation='vertical', label="Amplitud", ticklocation='left')  # Barra de color vertical
    # cbar_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    # Limpiar el subplot de frecuencias
    ax3.clear()
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Amplitud [mm]")
    ax3.set_xlim(times[0], times[-1])
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
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
    line, = ax3.plot(t_spec, amplitude_evolution, label=f"{frequency:.2f} Hz", marker=".")
    plotted_frequencies[frequency] = line
    ax3.legend()

# Callback para procesar texto ingresado
def on_submit_frequency(text):
    try:
        frequencies = [float(f.strip()) for f in text.split(",")]
        for freq in frequencies:
            plot_frequency_evolution(freq)
        fig_main.canvas.draw_idle()
    except ValueError:
        print("Por favor ingresa frecuencias válidas separadas por comas.")

# Callbacks para actualizar parámetros del espectrograma
def on_submit_nperseg(text):
    global nperseg
    try:
        nperseg = int(text)
        update_spectrogram()
        fig_main.canvas.draw_idle()
    except ValueError:
        print("Por favor ingresa un valor válido para nperseg.")

def on_submit_noverlap(text):
    global noverlap
    try:
        noverlap = int(text)
        update_spectrogram()
        fig_main.canvas.draw_idle()
    except ValueError:
        print("Por favor ingresa un valor válido para noverlap.")

def save_figures(val):
    file_name_important = file_name.split("/")[-1]
    file_name_important = "_".join(file_name_important.split("_")[:-2])
    now = str(datetime.datetime.now()).split(".")[-1]
    plt.savefig(f"spectrograms/{file_name_important}_{now}.png")
    plt.savefig(f"spectrograms/_{file_name_important}_{now}.pdf")


# Subplot 1: Serie temporal
ax1.plot(times, centers * 1e3) 
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Altura [mm]")
ax1.set_title("Serie Temporal")


# Cuadros de texto
text_box_freq_ax = fig_main.add_axes([0.15, 0.04, 0.2, 0.05])  # [left, bottom, width, height]
text_box_freq = TextBox(text_box_freq_ax, "Frecuencias (Hz): ")
text_box_freq.on_submit(on_submit_frequency)

text_box_nperseg_ax = fig_main.add_axes([0.45, 0.04, 0.2, 0.05])
text_box_nperseg = TextBox(text_box_nperseg_ax, "nperseg: ")
text_box_nperseg.on_submit(on_submit_nperseg)

text_box_noverlap_ax = fig_main.add_axes([0.75, 0.04, 0.2, 0.05])
text_box_noverlap = TextBox(text_box_noverlap_ax, "noverlap: ")
text_box_noverlap.on_submit(on_submit_noverlap)

button_ax = fig_main.add_axes([0.8455795, 0.945, 0.152, 0.05])
button = Button(button_ax, "Guardar imagen")
button.on_clicked(save_figures)


manager = plt.get_current_fig_manager()
manager.window.state('zoomed') 


# Mostrar figura
plt.show()
