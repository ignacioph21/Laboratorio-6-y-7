from skimage.io import imread, imsave
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
images_folder = dir_path.joinpath("Images")
output_folder =dir_path.joinpath("Output")

promediado = np.zeros((1024,1024))

i = 1
max_iter = 500
for file in os.listdir("./Gitignore/Referencia 2_202405_1537"): 
    if file[-3:] != "cih":
        if i > max_iter:
            break
        i += 1
        promediado += imread("./Gitignore/Referencia 2_202405_1537" + os.sep + file, as_gray=True)
        #plt.imshow(imread("./Gitignore/Gota 2_202405_1536/" + file, as_gray=True))
        #plt.show()

promediado /= i

plt.imshow(promediado)
plt.show()

x, y, w, h = 245, 475, 125, 125
imsave(images_folder.joinpath("ref-promediada-corte-cuadrado.tiff"), (promediado[y:y+h, x:x+h]).astype(np.uint8))
