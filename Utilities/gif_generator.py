import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def select_roi(image):
    # Convertir la imagen de PIL a un formato que OpenCV puede usar
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Mostrar la imagen y permitir la selección del ROI
    roi = cv2.selectROI("Selecciona el ROI", image_cv, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecciona el ROI")
    return roi

def crop_image(image, roi):
    x, y, w, h = roi
    return image.crop((x, y, x + w, y + h))

def resize_image(image, scale_factor):
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, Image.ANTIALIAS)

def create_gif(image_folder, start_frame, end_frame, output_file, scale_factor, frame_duration):
    # Incluir BMP en la lista de formatos de imagen
    supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    images = [f for f in os.listdir(image_folder) if f.endswith(supported_formats)]
    images.sort()
    
    if start_frame < 0 or end_frame >= len(images) or start_frame > end_frame:
        print("Los índices de los frames están fuera del rango.")
        return
    
    images_to_use = images[start_frame:end_frame+1]
    
    if not images_to_use:
        print("No se encontraron imágenes para el rango especificado.")
        return
    
    frames = []
    
    first_image_path = os.path.join(image_folder, images_to_use[0])
    first_image = Image.open(first_image_path)
    roi = select_roi(first_image)
    
    for image_name in images_to_use:
        image_path = os.path.join(image_folder, image_name)
        with Image.open(image_path) as img:
            if roi:
                img = crop_image(img, roi)
            if scale_factor:
                img = resize_image(img, scale_factor)
            frames.append(np.array(img))  # Convertir la imagen a un array de NumPy
    
    if frames:
        fig, ax = plt.subplots()
        ax.axis('off')
        ims = []
        
        for frame in frames:
            im = ax.imshow(frame, animated=True, cmap="gray")
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=frame_duration, blit=True, repeat_delay=1000)
        ani.save(output_file, writer='pillow', fps=1000/frame_duration)
        print(f"GIF creado: {output_file}")
    else:
        print("No se encontraron imágenes para crear el GIF.")

def main():
    root = Tk()
    root.withdraw()
    
    folder_selected = filedialog.askdirectory(title="Selecciona la carpeta con imágenes")
    if not folder_selected:
        print("No se seleccionó ninguna carpeta.")
        return
    
    images = [f for f in os.listdir(folder_selected) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    images.sort()
    
    if not images:
        print("No hay imágenes en la carpeta seleccionada.")
        return
    
    start_frame = int(input(f"Introduce el número del frame inicial (0-{len(images)-1}): "))
    end_frame = int(input(f"Introduce el número del frame final (0-{len(images)-1}): "))
    
    if start_frame < 0 or end_frame >= len(images) or start_frame > end_frame:
        print("Los índices de los frames están fuera del rango.")
        return
    
    scale_factor = simpledialog.askfloat("Escala de imagen", "Introduce el factor de escala (ej. 0.5 para la mitad):", minvalue=0.01, maxvalue=10.0)
    
    frame_duration = simpledialog.askinteger("Duración entre Frames", "Introduce la duración entre frames en milisegundos:", minvalue=1)
    
    output_file = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF files", "*.gif")], title="Guardar GIF como")
    if not output_file:
        print("No se seleccionó un nombre de archivo para guardar.")
        return
    
    create_gif(folder_selected, start_frame, end_frame, output_file, scale_factor, frame_duration)

if __name__ == "__main__":
    main()
