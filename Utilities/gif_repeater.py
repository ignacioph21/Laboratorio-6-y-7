import os
from tkinter import Tk, filedialog
from PIL import Image, ImageSequence

# Función para seleccionar un archivo GIF
def select_gif_file():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
    return file_path

# Función para modificar un GIF para que se repita
def make_gif_loop(file_path):
    if file_path:
        im = Image.open(file_path)

        frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
        directory, filename = os.path.split(file_path)
        new_filename = os.path.join(directory, "looped_" + filename)

        frames[0].save(new_filename, save_all=True, append_images=frames[1:], loop=0)

        print(f"GIF modificado guardado como: {new_filename}")
    else:
        print("No se seleccionó ningún archivo.")

# Seleccionar el archivo GIF y hacer que se repita
gif_file = select_gif_file()
make_gif_loop(gif_file)
