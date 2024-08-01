from tkinter.filedialog import askdirectory
from tkinter import Tk
import numpy as np
import cv2
import os

# Función para leer el archivo .cih y extraer la información necesaria
def read_cih_file(cih_file_path):
    with open(cih_file_path, 'r') as file:
        lines = file.readlines()
    
    info = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip()] = value.strip()
    
    return info

# Selección del directorio de imágenes
root = Tk()
root.withdraw()  # Esconder la ventana principal de Tkinter
image_pattern = askdirectory(title='Select Folder')
root.destroy()
print(f"Directorio seleccionado: {image_pattern}")

# Buscar el archivo .cih en el directorio seleccionado
cih_file_name = os.path.basename(image_pattern) + '.cih'
cih_file_path = os.path.join(image_pattern, cih_file_name)

if not os.path.exists(cih_file_path):
    raise FileNotFoundError(f"El archivo {cih_file_name} no se encuentra en la carpeta seleccionada: {image_pattern}")

# Cargar la información del archivo .cih
cih_info = read_cih_file(cih_file_path)

# Extraer los valores necesarios
fps = int(cih_info.get('Record Rate(fps)', 1000))
start = int(cih_info.get('Start Frame', 1))
image_width = int(cih_info.get('Image Width', 1024)) // 2  # Reduciendo la resolución a la mitad
image_height = int(cih_info.get('Image Height', 1024)) // 2

# Obtener la lista de imágenes
images = sorted(os.listdir(image_pattern))

# Filtrar las imágenes válidas
valid_images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.bmp'))]

if not valid_images:
    raise FileNotFoundError("No se encontraron imágenes válidas en la carpeta seleccionada.")

# Inicialización de la ventana de video
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", image_width, image_height)

frame_num = start

while frame_num < start + len(valid_images):
    image_index = frame_num - start  # Corregir el índice de la imagen
    if image_index >= len(valid_images):
        # print(f"El índice {image_index} está fuera de rango para valid_images.")
        break

    image_path = os.path.join(image_pattern, valid_images[image_index])
    # print(f"Cargando imagen: {image_path}")  # Mensaje de depuración
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        # print(f"Error al cargar la imagen: {image_path}")
        break

    image = cv2.resize(image, (image_width, image_height))
    
    # Mostrar el frame y el número del frame
    cv2.imshow("Video", image)
    cv2.setWindowTitle("Video", f"Frame: {frame_num}")

    # Controlar la velocidad del video
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break
    
    frame_num += 1

cv2.destroyAllWindows()
