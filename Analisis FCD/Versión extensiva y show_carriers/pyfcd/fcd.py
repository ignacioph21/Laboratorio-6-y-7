import time
from dataclasses import dataclass
from typing import List
import os.path

import imageio.v2 as imageio
import numpy as np
from numpy import array
from scipy.fft import fft2, ifft2, ifftshift
from skimage.draw import disk
from skimage.io import imread
from more_itertools import flatten

from pyfcd.fft_inverse_gradient import fftinvgrad
from pyfcd.find_peaks import find_peaks
from pyfcd.kspace import pixel2kspace
import matplotlib.pyplot as plt


def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())

def peak_mask(shape, pos, r):
    result = np.zeros(shape, dtype=bool)
    result[disk(pos, r, shape=shape)] = True
    return result


def ccsgn(i_ref_fft, mask):
    return np.conj(ifft2(i_ref_fft * mask))


@dataclass
class Carrier:
    pixel_loc: array
    k_loc: array
    krad: float
    mask: array
    ccsgn: array


def calculate_carriers(i_ref, show_carriers = False):
    peaks = find_peaks(i_ref)
    peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
    i_ref_fft = fft2(i_ref)

    if show_carriers:
        plt.imshow(i_ref, cmap='gray') # 
        # Supongamos que tienes tus vectores k almacenados en las variables k1 y k2
        k1 = pixel2kspace(i_ref.shape, peaks[0])
        k2 = pixel2kspace(i_ref.shape, peaks[1])

        # Calcula los puntos de inicio y fin de la línea en la imagen
        start_point = peaks[0]  # Punto de inicio en la imagen
        end_point = peaks[0] + 50*k1  # Punto final en la dirección del vector k1
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color='red')

        start_point = peaks[0+1]  # Punto de inicio en la imagen
        end_point = peaks[0+1] + 50*k2 # 1  # Punto final en la dirección del vector k1
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color='red')

        plt.show()


    carriers = [Carrier(peak, pixel2kspace(i_ref.shape, peak), peak_radius, mask, ccsgn(i_ref_fft, mask)) for mask, peak
                in
                [(ifftshift(peak_mask(i_ref.shape, peak, peak_radius)), peak) for peak in peaks]]
    return carriers


def fcd(i_def, carriers: List[Carrier]):
    i_def_fft = fft2(i_def)

    phis = [-np.angle(ifft2(i_def_fft * c.mask) * c.ccsgn) for c in carriers]

    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a
    v = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a

    return fftinvgrad(-u, -v)

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=Path)
    argparser.add_argument('reference_image', type=str)
    argparser.add_argument('definition_image', nargs='+', help='May contain wildcards')
    argparser.add_argument('--output-format', default='tiff', choices=['tiff', 'bmp', 'png', 'jpg', 'jpeg'], help='The output format')
    argparser.add_argument('--skip-existing', action='store_true', help='Skip processing an image if the output file already exists')

    args = argparser.parse_args()

    args.output_folder.mkdir(exist_ok=True)

    i_ref = imread(args.reference_image, as_gray=True)
    print(max(i_ref[0]), min(i_ref[0]))

    print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    print('done')

    files = list(sorted(flatten((glob.glob(x) if '*' in x else [x]) for x in args.definition_image)))

    for file in files:
        output_file_path = args.output_folder.joinpath(f'{Path(file).stem}.{args.output_format}')

        if os.path.abspath(file).lower() == os.path.abspath(output_file_path).lower():
            print(f'Warning: Skipping converting {file} because it would overwrite a input file')
            continue

        if args.skip_existing and output_file_path.exists():
            continue

        print(f'processing {file} -> {output_file_path} ... ', end='')
        i_def = imread(file, as_gray=True)
        t0 = time.time()
        height_field = fcd(i_def, carriers)
        print(f'done in {time.time() - t0:.2}s\n')

        imageio.imwrite(output_file_path, (normalize_image(height_field) * 255.0).astype(np.uint8))
