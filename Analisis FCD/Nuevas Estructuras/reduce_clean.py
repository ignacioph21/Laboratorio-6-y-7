import numpy as np
import matplotlib.pyplot as plt

# Cargar el array de numpy
heightfields = np.load('heightfields.npy')

# Parámetros de la señal
fps = 500  # Frecuencia de muestreo en Hz
dt = 1 / fps  # Intervalo de tiempo entre frames
frecuencia_deseada = 3.38  # Frecuencia de interés en Hz

# Dimensiones del array
n_frames, height, width = heightfields.shape

# Eje de frecuencia para la FFT
frequencies = np.fft.fftfreq(n_frames, d=dt)
idx_frecuencia_deseada = np.argmin(np.abs(frequencies - frecuencia_deseada))

# Calcular la FFT en cada punto y obtener la amplitud de la frecuencia deseada
amplitudes = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        # Calcular la FFT a lo largo del tiempo en la posición (i, j)
        fft_values = np.fft.fft(heightfields[:, i, j])
        amplitudes[i, j] = np.abs(fft_values[idx_frecuencia_deseada])

# Guardar el array de amplitudes en un archivo .npy
np.save(f'amplitudes_frecuencia_{frecuencia_deseada}Hz.npy', amplitudes)  

# Mostrar las amplitudes con imshow
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Gráfico de imshow de las amplitudes
im = ax1.imshow(amplitudes, cmap='viridis')
fig.colorbar(im, ax=ax1, label='Amplitud')
ax1.set_title(f'Amplitud en la frecuencia de {frecuencia_deseada} Hz')
ax1.set_xlabel('Ancho (píxeles)')
ax1.set_ylabel('Alto (píxeles)')

# Verificación: FFT en el punto central
central_i, central_j = height // 2, width // 2
fft_central = np.fft.fft(heightfields[:, central_i, central_j])
amplitud_fft_central = np.abs(fft_central)

# Gráfico de la FFT en el punto central
ax2.plot(frequencies[:n_frames // 2], amplitud_fft_central[:n_frames // 2])
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitud')
ax2.set_title('FFT en el punto central')
ax2.grid()

plt.tight_layout()
plt.show()
