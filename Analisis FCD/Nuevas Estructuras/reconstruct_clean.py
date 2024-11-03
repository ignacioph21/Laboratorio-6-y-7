import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cargar las amplitudes guardadas
amplitudes = np.load('amplitudes_frecuencia_3.38Hz.npy')

# Parámetros de la animación
fps = 500  # Frecuencia de muestreo en Hz (asegúrate de que coincide con la frecuencia original de tus datos)
f = 3.38   # Frecuencia de oscilación deseada en Hz (puedes cambiar este valor)
dt = 1 / fps  # Intervalo de tiempo entre frames
t_max = 2  # Duración en segundos de la animación
n_frames = int(t_max * fps)  # Número total de frames

# Crear la figura para la animación
fig, ax = plt.subplots()
im = ax.imshow(amplitudes * np.sin(2 * np.pi * f * 0), cmap='viridis', vmin=-amplitudes.max(), vmax=amplitudes.max())
plt.colorbar(im, label='Amplitud')

# Función de actualización para la animación
def update(frame):
    t = frame * dt  # Tiempo actual
    oscillation = amplitudes * np.sin(2 * np.pi * f * t)
    im.set_array(oscillation)
    return [im]

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)

# Mostrar la animación
plt.title(f'Oscilaciones reconstruidas a {f} Hz')
plt.xlabel('Ancho (píxeles)')
plt.ylabel('Alto (píxeles)')
plt.show()
