#%% PACKAGES
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from pyfcd.fcd import calculate_carriers, fcd
from scipy.signal.windows import tukey
from scipy.signal import find_peaks
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, warp_polar
from tqdm import tqdm

#%% SETUP
start_frame = 2700
stop_frame  = 2800

#%% UPLOADING
images_path = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("Imagenes")

i_ref_path = images_path.joinpath("0611-gota_con_jeringa/i_ref.bmp")
i_def_paths = sorted([p for p in images_path.glob("0611-gota_con_jeringa/i_defs/*.bmp")])[start_frame:stop_frame]

flag = cv2.IMREAD_UNCHANGED
i_ref = cv2.imread(str(i_ref_path), flag)
i_defs = [cv2.imread(str(p), flag) for p in i_def_paths]
plt.imshow(i_defs[0])

#%% AUXILIARY FUNCTIONS
def masked(i_def, i_ref, mask):
    return i_def*mask + i_ref*(1-mask)

def find_circle(data, radii, low=None, high=None, sigma=1.0, mask=None, show_edges=False, **canny_kwargs):
    edges = canny(data, sigma=sigma, low_threshold=low, high_threshold=high, use_quantiles=True, mask=mask)
    
    if show_edges:
        plt.imshow(edges)
        plt.show()

    hough_res = hough_circle(edges, radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
    return np.array([cx[0], cy[0]]), radii

def beautify_axs(ax, title="", labels=["",""], multiply=None, legend_kwargs=None, grid=False, \
                 aspect="auto", xlim=None, ylim=None, margins=(0,0)):

    ax.set_title(title, ha='left', position=(0, 1))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.margins(*margins)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    if multiply is not None:
        xticks = np.array(ax.get_xticks())
        yticks = np.array(ax.get_yticks())
        ax.set_xticks(xticks, labels=np.round(xticks*multiply[0], 2))
        ax.set_yticks(yticks, labels=np.round(yticks*multiply[1], 2))

    ax.set_aspect(aspect)
    if legend_kwargs is not None: ax.legend(**legend_kwargs)
    if grid: ax.grid()

    return None

#%% FCD ANALYSIS
# PARÁMETERS
hstar = 0.0323625     # [m] #TODO: revisar
square_size = 2.2e-3  # [m]

# FRAME PREPROCESSING
mask = np.ones_like(i_ref)
mask[:500,800:] = 0

roi = 0, 0, *i_ref.shape
window1dx = np.abs(tukey(roi[-1], 0.1))
window1dy = np.abs(tukey(roi[-2], 0.1))
window2d = np.sqrt(np.outer(window1dx, window1dy))

# ANALYSIS
carriers = calculate_carriers(i_ref*window2d, square_size=square_size)
height_maps = [fcd(masked(i_def, i_ref, mask)*window2d, carriers, h=hstar, unwrap=True) for i_def in tqdm(i_defs)]
PXtoM = carriers[0].PXtoM

#%% FIND DROP CENTER
radii = np.arange(300, 450, 5)    # a ojo
c, R = find_circle(height_maps[0], radii=radii, low=0.8, high=0.9, sigma=10, mask=mask.astype(bool), show_edges=True)

print("centro:", c)
print("radio:", R)

# PLOT
plt.figure()
ax = plt.subplot()

circle = plt.Circle(c, R, color='r', fill=False)

ax.pcolormesh(height_maps[0], cmap="Greys")
ax.add_patch(circle)
ax.scatter(*c, marker="s", color="r")
ax.set_aspect("equal")
plt.show()

# %%
radius = int(np.abs(c[0]))

i_polars = np.zeros((len(i_defs), radius))
for i, img in enumerate(height_maps):
    i_polar = warp_polar(img, radius=radius, center=c[::-1])
    i_polars[i] = np.mean(i_polar[150:210], axis=0)
    i_polars[i] = i_polars[i] 

    if i == 0: 
        plt.figure()
        ax = plt.subplot()
        arc = mpl.patches.Arc(c*PXtoM*100, 2*radius*PXtoM*100, 2*radius*PXtoM*100, theta1=140, theta2=210, lw=1, ls="--", edgecolor="r", zorder=2)
        ax.pcolormesh(np.arange(img.shape[0])*PXtoM*100, np.arange(img.shape[1])*PXtoM*100, img, cmap="Greys")
        ax.add_patch(arc)
        ax.scatter(*c*PXtoM*100, marker="s", color="r")
        beautify_axs(ax, labels=[r"$x$ [cm]",r"$y$ [cm]"], aspect="equal", multiply=[100,100])
        plt.show()

        plt.figure()
        ax = plt.subplot()
        i_polar = warp_polar(img, radius=radius, center=c[::-1])
        ax.pcolormesh(np.arange(radius)*PXtoM*100, np.arange(0,360), i_polar, cmap="Greys", shading="nearest")
        ax.axhline(210, c="r", ls="--")
        ax.axhline(150, c="r", ls="--")
        beautify_axs(ax, labels=[r"$r$ [cm]", "r$\theta$ [deg]"])
        plt.show()

t = np.arange(i_polars.shape[0])/500
r = np.arange(i_polars.shape[1])*PXtoM

plt.figure(dpi=200)
ax = plt.subplot()

im = ax.pcolormesh(r*100, t*1000, i_polars, cmap="seismic", norm=TwoSlopeNorm(0))

plt.xlabel("$r$ [cm]")
plt.ylabel("$t$ [ms]")
plt.show()