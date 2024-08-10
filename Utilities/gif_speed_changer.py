import imageio
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def change_gif_fps(input_gif_path, new_fps):
    # Leer el archivo GIF en partes
    reader = imageio.get_reader(input_gif_path)
    
    # Calcular la duración por cuadro basada en el nuevo FPS
    new_duration_per_frame = 1000 / new_fps  # En milisegundos
    
    # Crear una lista de nuevos cuadros con la nueva duración
    new_gif = []
    for frame in reader:
        new_gif.append(imageio.core.util.Image(frame, meta={'duration': new_duration_per_frame}))

    reader.close()

    # Guardar el nuevo GIF con un nombre diferente
    output_gif_path = os.path.splitext(input_gif_path)[0] + f'_fps_{new_fps}.gif'
    imageio.mimsave(output_gif_path, new_gif, duration=new_duration_per_frame/1000)

    print(f'Nuevo archivo GIF guardado como: {output_gif_path}')

def select_file_and_change_fps():
    # Crear una ventana oculta de Tkinter
    root = Tk()
    root.withdraw()

    # Seleccionar archivo GIF
    input_gif_path = askopenfilename(filetypes=[("GIF files", "*.gif")])
    if not input_gif_path:
        print("No se seleccionó ningún archivo.")
        return

    # Pedir los FPS al usuario
    new_fps = float(input("Ingrese los nuevos frames por segundo (FPS): "))

    # Cambiar la velocidad del GIF
    change_gif_fps(input_gif_path, new_fps)

# Ejecutar la función para seleccionar el archivo y cambiar los FPS
select_file_and_change_fps()
