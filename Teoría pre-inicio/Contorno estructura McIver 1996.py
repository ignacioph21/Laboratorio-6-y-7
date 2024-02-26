import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1

K = 1

def psi(x, y):
    return np.exp(-K*y+1j*K*x)*(np.pi*np.sign(K*x-np.pi/2)-np.pi*np.sign(K*x+np.pi/2) + 1j*exp1(-K*y+1j*K*x+1j*np.pi/2)-1j*exp1(-K*y+1j*K*x-1j*np.pi/2))

xs = np.linspace(-5, 5, 1000)
ys = np.linspace(-1, 3, 1000)

X, Y = np.meshgrid(xs, ys)

Phi = psi(X, Y).imag

plt.contour(X, Y, Phi, 250)
plt.gca().invert_yaxis()
plt.hlines(0, min(xs), max(xs))
plt.show()
