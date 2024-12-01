from scipy.fft import fftfreq, fft, rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
import h5py

plt.rc('font', size=16-2)

def video_fft(frames, fps=1, apply_fftshift=False):
    frames_fft = rfft(frames, axis=0)
    freqs = rfftfreq(len(frames), 1/fps)
    return freqs, frames_fft

def amplitude(freqs, frames_fft, fps=1, freq_loc=None, sigmas=1):
    differences = np.abs(np.abs(freqs) - freq_loc)
    index = np.argwhere(differences == min(differences))[0][0]
    print(f"Diferencia con la frecuencia deseada es de: {differences[index]:.2e} Hz.")
    return np.abs(frames_fft[index]), np.angle(frames_fft[index])

def on_submit(text):
##    try:
        # Procesar frecuencias ingresadas
        fs = [float(f) for f in text.split(",")]
        for f in fs:
            amplitudes, fases = amplitude(freqs, vidfft, fps=125, freq_loc=f) 
            np.save(f"outputs/amplitude_{f}Hz_{file_name[:-3]}_{ranges[0]}_{ranges[1]}", amplitudes)
            np.save(f"outputs/fases_{f}Hz_{file_name[:-3]}_{ranges[0]}_{ranges[1]}", fases)

            # Mostrar resultado de amplitud
            fig, axs = plt.subplots(1, 2)
            plt.suptitle(f"Frecuencia: {f} Hz.")
            im1 = axs[0].imshow(amplitudes * 1e3, cmap="Grays")
            plt.colorbar(im1, ax=axs[0], label="Amplitud [mm]")
            axs[0].add_patch(plt.Circle((c, c), 180, color='red', fill=False))

            im2 = axs[1].imshow(fases, cmap="hsv")            
            plt.colorbar(im2, ax=axs[1], label="Fases")
            axs[1].add_patch(plt.Circle((c, c), 180, color='red', fill=False))

            fig.set_size_inches(16, 9)

            plt.savefig(f"outputs/{file_name[:-3]}_{f}Hz_{ranges[0]}_{ranges[1]}.png")
            plt.savefig(f"outputs/{file_name[:-3]}_{f}Hz_{ranges[0]}_{ranges[1]}.pdf")

            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed') 

            plt.show()
##    except ValueError:
##        print("Por favor, introduce frecuencias válidas separadas por comas.")

# Configuración inicial
file_name = "F:/Ignacio Hernando/Mediciones Procesadas/Toroide Grande/Exponencial_grande_2_202411_1112.h5"
ranges = (300, 1200)

# Leer datos desde el archivo HDF5
with h5py.File(file_name, 'r') as f:
    c = f['data'][0].shape[0] // 2
    centers = f['data'][ranges[0]:ranges[1], c, c]
    frames = f['data'][ranges[0]:ranges[1]] #  
    
# Calcular FFT del video
freqs, vidfft = video_fft(frames, fps=125, apply_fftshift=True) # 

# Calcular tiempos
times = np.arange(len(centers)) / 125 * 1000

# Crear figura para la visualización
fig_main = plt.figure()
            

# Subplot 1: Altura vs Frames
plt.subplot(211)
plt.plot(np.arange(ranges[0], ranges[1]), centers * 1e3)
plt.xlabel("Frames")
plt.ylabel("Altura [mm]")

# Subplot 2: FFT
plt.subplot(212)
freqs = fftfreq(len(times), times[1] - times[0]) * 1000
ffts = fft((centers-np.mean(centers)) * 1e3)
plt.plot(freqs[:len(times)//2][::-1], np.abs(ffts)[:len(times)//2][::-1], marker=".")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [mm]")

# Crear un TextBox para ingresar frecuencias
axbox = plt.axes([0.15, 0.01, 0.7, 0.05])  # [left, bottom, width, height]
text_box = TextBox(axbox, "Frecuencias (Hz): ")
text_box.on_submit(on_submit)

# Ajustar el espacio para el TextBox
plt.subplots_adjust(bottom=0.2, hspace=0.271, top=0.96)

fig_main.set_size_inches(16, 9)

file_name = file_name.split("/")[-1]
plt.savefig(f"outputs/Oscilaciones_{file_name[:-3]}_{ranges[0]}_{ranges[1]}.png")
plt.savefig(f"outputs/Oscilaciones_{file_name[:-3]}_{ranges[0]}_{ranges[1]}.pdf")

manager = plt.get_current_fig_manager()
manager.window.state('zoomed') 


plt.show() 

