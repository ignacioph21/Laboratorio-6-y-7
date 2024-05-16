import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift, fftfreq
from matplotlib.colors import LogNorm  # Importar LogNorm para escala logarítmica
from skimage.restoration import unwrap_phase

from pyfcd.fft_inverse_gradient import fftinvgrad
from pyfcd.find_peaks import find_peaks
from pyfcd.kspace import pixel2kspace

def selectSquareROI(window_name, img):
    # Utiliza selectROI para seleccionar un ROI rectangular
    rect = cv2.selectROI(window_name, img)
    
    # Extrae las coordenadas y dimensiones del ROI
    x, y, w, h = rect
    
    # Encuentra la dimensión más pequeña (ancho o alto)
    size = min(w, h)
    
    # Ajusta el ROI seleccionado para que sea un cuadrado
    x_adjusted = x + (w - size) // 2
    y_adjusted = y + (h - size) // 2
    square_roi = (x_adjusted, y_adjusted, size, size)
    
    return square_roi

def plot_fft(i_ref, i_def, N=5):
    """
    Plotea la transformada de Fourier de las imágenes de referencia y deformada,
    junto con los círculos que indican los picos de interés.
    """

    peaks = find_peaks(i_ref)
    peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2

    # Cálculo de la transformada de Fourier y ajuste del espectro
    fft_ref = fftshift(np.abs(fft2(i_ref - np.mean(i_ref))))
    fft_def = fftshift(np.abs(fft2(i_def - np.mean(i_def))))

    # Mostrar las imágenes en blanco y negro y en escala logarítmica
    plt.subplot(121)
    plt.imshow(fft_ref, cmap='gray', norm=LogNorm())  # LogNorm aplica escala logarítmica
    plt.title('Imagen de referencia')

    plt.subplot(122)
    plt.imshow(fft_def, cmap='gray', norm=LogNorm())
    plt.title('Imagen deformada')

    # Dibujar círculos en la imagen deformada para los dos picos
    for peak_center in peaks:
        circle = plt.Circle(peak_center[::-1], peak_radius, color='r', fill=False)  # Crear el círculo
        plt.gca().add_patch(circle)  # Agregar el círculo a la imagen

    rows, cols = np.shape(i_ref)

    k_space_rows = fftshift(np.fft.fftfreq(rows, 1 / (2.0 * np.pi)))
    k_space_cols = fftshift(np.fft.fftfreq(cols, 1 / (2.0 * np.pi)))

    x_ticks = np.linspace(0, cols-1, N, dtype=int)
    y_ticks = np.linspace(0, rows-1, N, dtype=int)
      
    x_k_ticks = k_space_rows[x_ticks]  # Convertir coordenadas x a espacio k
    y_k_ticks = k_space_cols[y_ticks]  # Convertir coordenadas y a espacio k

    plt.subplot(122) # TODO: beutify_axis(axis, ticks, labels).
    plt.xticks(x_ticks, [f'{k:.2f}' for k in x_k_ticks])
    plt.yticks(y_ticks, [f'{k:.2f}' for k in y_k_ticks])
    plt.xlabel("k_x")
    plt.ylabel("k_y")

    plt.subplot(121)
    plt.xticks(x_ticks, [f'{k:.2f}' for k in x_k_ticks])
    plt.yticks(y_ticks, [f'{k:.2f}' for k in y_k_ticks])
    plt.xlabel("k_x")
    plt.ylabel("k_y")

    plt.show()

def plot_with_arrows(i_ref, peaks):
    plt.imshow(i_ref, cmap='gray')  

    module = 0.25 * np.shape(i_ref)[0]
    
    # Supongamos que tienes tus vectores k almacenados en las variables k1 y k2
    k1 = pixel2kspace(i_ref.shape, peaks[0])
    k2 = pixel2kspace(i_ref.shape, peaks[1])

    k1 /= np.linalg.norm(k1)
    k2 /= np.linalg.norm(k2)

    # Calcula los puntos de inicio y fin de la línea en la imagen
    start_point = peaks[0]  # Punto de inicio en la imagen
    end_point = peaks[0] + module * np.array(k1)  # Punto final en la dirección del vector k1
    plt.arrow(start_point[1], start_point[0], end_point[1] - start_point[1], end_point[0] - start_point[0],
              head_width=10, head_length=10, fc='red', ec='red')
    plt.text(end_point[1]*1.1, end_point[0]*1.1, 'k1', color='red', fontsize=12, ha='left', va='center')

    start_point = peaks[1]  # Punto de inicio en la imagen
    end_point = peaks[1] + module * np.array(k2)  # Punto final en la dirección del vector k2
    plt.arrow(start_point[1], start_point[0], end_point[1] - start_point[1], end_point[0] - start_point[0],
              head_width=10, head_length=10, fc='red', ec='red')
    plt.text(end_point[1]*1.1, end_point[0]*1.1, 'k2', color='red', fontsize=12, ha='left', va='center')

    plt.axis('off')  # No mostrar ejes
    plt.show()

def plot_angles(angles_x, angles_y):
    plt.subplot(221)
    plt.title("Ángulos en dirección k_1.")
    plt.imshow(angles_x)
    plt.axis('off')  # No mostrar ejes

    plt.subplot(222)
    plt.title("Ángulos en dirección k_2.")
    plt.imshow(angles_y)
    plt.axis('off')  # No mostrar ejes

    plt.subplot(223)
    plt.title("Ángulos en dirección k_1 Unwrapped.")
    plt.imshow(unwrap_phase(angles_x))
    plt.axis('off')  # No mostrar ejes

    plt.subplot(224)
    plt.title("Ángulos en dirección k_2 Unwrapped.")
    plt.imshow(unwrap_phase(angles_y))
    plt.axis('off')  # No mostrar ejes

    plt.show()


def process_sliced(sliced):
    sign = 1 if abs(max(sliced)) > abs(min(sliced)) else -1
    sliced -= sliced[-1]
    sliced *= sign
    sliced /= max(sliced)
    return sliced

def plot_height_field(height_field, i_teo, roi):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))
    
    # Plot de la imagen centrada y ocupando 3/4 de la altura
    im = axs[0].imshow(height_field, aspect='auto')
    axs[0].hlines(height_field.shape[0] // 2, 0, height_field.shape[0+1], linestyle='--', linewidth=2, color='white')
    fig.colorbar(im, ax=axs[0])  # Agregar barra lateral
    axs[0].axis('off')  # Eliminar etiquetas de ejes x e y

    # Plot de la línea en el medio de la imagen
    sliced =  process_sliced(height_field[height_field.shape[0] // 2, :])
    axs[1].plot(sliced, label="Resultado FCD.") # No sé si esto es x o y.    

    if not (i_teo is None):
        i_teo = np.array(i_teo, dtype=np.float32)[roi[1]:roi[1]+roi[-1] , roi[0]:roi[0]+roi[-2]]
        i_teo_sliced = i_teo[i_teo.shape[0] // 2, :]
        i_teo_sliced = process_sliced(i_teo_sliced)
        axs[1].plot(i_teo_sliced, label="Alturas teóricas.")
        axs[1].legend()
    
    plt.tight_layout()
    plt.show()
