import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pyfcd.fcd import calculate_carriers, fcd, normalize_image
from pathlib import Path
from skimage.io import imread, imsave

images_folder = Path("Análisis FCD" + os.sep + "Images")
output_folder = Path("Análisis FCD" + os.sep + "Output")
reference_image = images_folder.joinpath("test image reference.tiff")
definition_images = ["test image displaced.tiff"]
roi = ()

i_ref = imread(reference_image, as_gray=True)
for file in definition_images:
    image_file_path = images_folder.joinpath(file)
    output_file_path = output_folder.joinpath(f'output-{Path(file).stem}.tiff')

    i_def = imread(image_file_path, as_gray=True)
    (x,y,w,h) = cv2.selectROI(i_def) if roi == () else roi
    print("roi:", roi) # por si queremos volver a seleccionar la misma región

    i_ref = np.array(i_ref, dtype=np.float32)
    i_def = np.array(i_def, dtype=np.float32)

    i_ref_ = i_ref[y:y+h , x:x+w]
    i_def = i_def[y:y+h , x:x+w]

    i_ref -= np.mean(i_ref)
    i_def -= np.mean(i_def)

    print(f'processing reference image...', end='') #TODO: Cambiar el texto.
    carriers = calculate_carriers(i_ref_) #TODO: No entiendo porque le restamos el promedio a i_ref si despues no lo usamos
    print('done')

    t0 = time.time()
    height_field = fcd(i_def, carriers)
    print(f'done in {time.time() - t0:.2}s\n')
    plt.imshow(height_field)
    plt.show()    

    (x,y,w,h) = cv2.selectROI((normalize_image(height_field) * 255.0).astype(np.uint8)) 
    fondo = np.mean(height_field[y:y+h,:], axis=0)
    plt.plot(fondo)
    plt.show()

    imsave(output_file_path, height_field)

