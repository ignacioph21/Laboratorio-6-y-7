import matplotlib.pyplot as plt
import matplotlib.animation as anm
import scipy.special as sp
import numpy as np
from IPython.display import HTML


N = 50
NT = 24 

a = 2
k = 1
K = 0.1
w = np.sqrt(K)
h = -np.arctanh(K/k)/k

z = 0

N_0 = np.sqrt(0.5*(1+np.sinc(2j*k)))
def psi_0(z):
    return 1/N_0 * np.cosh(k*(z+h))

def eps(m):
    return 2 if m else 1

def phi(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = -np.arctan2(y,x) 

    total = 0
    for m in range(0, N):
        total += eps(m) * 1j**m * (sp.jv(m, k*r)-sp.jvp(m, k*a)*sp.hankel1(m, k*r)/sp.h1vp(m, k*a)) * np.cos(m*theta)
    return total


def phi_I(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = -np.arctan2(y,x)

    return np.exp(1j*k*r*np.cos(theta)) 

def phi_D(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = -np.arctan2(y,x) 

    total = 0
    for m in range(0, N):
        total += -eps(m) * 1j**m * sp.jvp(m, k*a)*sp.hankel1(m, k*r)/sp.h1vp(m, k*a) * np.cos(m*theta)
    return total


""" Gráfico """
# Coordenadas (de cartesianas a polares para que funciones el StreamPlot)
xs = np.linspace(-10, 10, 100)
ys = np.linspace(-10, 10, 100)
ts = np.linspace(0, 2*np.pi/w, NT)

X, Y, T = np.meshgrid(xs, ys, ts)

mask = np.sqrt(X**2 + Y**2) > a

# Evaluar el potencial (dentro del cilindro 0)
Phi = phi(X, Y) * psi_0(z) * np.exp(-1j*w*T) 
Phi *= mask
Phi = Phi.real

# # Gradiente
# gradient = np.gradient(Phi, xs, ys, ts)
# ux = gradient[1].real
# uy = gradient[0].real


fig, ax = plt.subplots()
cax = ax.pcolormesh(X[:, :, 0], Y[:, :, 0], Phi[:, :, 0])
fig.colorbar(cax)

def animate(i):
   cax.set_array(Phi[:, :, i].flatten())

a = anm.FuncAnimation(fig, animate, interval=100, frames=len(ts) - 1)
HTML(a.to_html5_video())
