import imageio
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def reduce_gif_frames(input_gif_path, frame_interval):
    # Leer el archivo GIF en partes
    reader = imageio.get_reader(input_gif_path)
    
    # Crear una lista de cuadros, conservando solo 1 de cada x frames
    new_gif = []
    for i, frame in enumerate(reader):
        if i % frame_interval == 0:
            new_gif.append(frame)
    
    reader.close()

    # Guardar el nuevo GIF con un nombre diferente
    output_gif_path = os.path.splitext(input_gif_path)[0] + f'_every_{frame_interval}_frames.gif'
    imageio.mimsave(output_gif_path, new_gif, duration=reader.get_meta_data().get('duration', 100) / 1000)

    print(f'Nuevo archivo GIF guardado como: {output_gif_path}')

def select_file_and_reduce_frames():
    # Crear una ventana oculta de Tkinter
    root = Tk()
    root.withdraw()

    # Seleccionar archivo GIF
    input_gif_path = askopenfilename(filetypes=[("GIF files", "*.gif")])
    if not input_gif_path:
        print("No se seleccionó ningún archivo.")
        return

    # Pedir el intervalo de frames al usuario
    frame_interval = int(input("Ingrese el intervalo de frames (por ejemplo, 2 para conservar 1 de cada 2 frames): "))

    # Reducir los frames del GIF
    reduce_gif_frames(input_gif_path, frame_interval)

# Ejecutar la función para seleccionar el archivo y reducir los frames
select_file_and_reduce_frames()
