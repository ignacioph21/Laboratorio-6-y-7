import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Nombre del archivo y rango de frames
file_name = "Exponencial_grande_2_202411_1112.h5"
ranges = (0, 1200)  # Rango de frames a animar

# Cargar los datos del archivo HDF5
with h5py.File(file_name, 'r') as f:
    data = f['data'][ranges[0]:ranges[1]] * 1e3  # Selección del rango de frames

# Configuración de la figura
fig, ax = plt.subplots()
frame_im = ax.imshow(data[0], cmap="viridis", interpolation="none")
cbar = plt.colorbar(frame_im, ax=ax, label="Amplitud [mm]")  # Barra de color
title = ax.set_title(f"Frame {ranges[0]}")  # Título inicial
ax.axis("off")

# Función para actualizar los datos en cada frame
def update(frame):
    frame_data = data[frame]
    frame_data -= frame_data[0, 0]

    frame_im.set_array(frame_data)
    
    # Actualizar el título con el número de frame
    title.set_text(f"Frame {ranges[0] + frame}")
    
    # Actualizar límites de color en función de los datos
    vmin, vmax = frame_data.min(), frame_data.max()
    frame_im.set_clim(vmin, vmax)  # Actualizar límites de color de la imagen
    cbar.update_normal(frame_im)  # Actualizar la barra de color para reflejar los nuevos límites
    
    return frame_im, title

# Crear la animación
ani = FuncAnimation(
    fig, 
    update, 
    frames=np.arange(len(data)), 
    interval=50,  # Duración de cada frame en ms
    blit=False  # Desactivamos `blit` para garantizar la actualización completa
)

plt.show()
