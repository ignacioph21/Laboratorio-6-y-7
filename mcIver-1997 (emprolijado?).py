import scipy.special as sp
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def En(n,z): 
    """
    Generalized exponential integral E_n for complex arguments.
    """
    if n <= 0:
        raise Exception("n must be greater than 0.")
    elif n == 1:
        return sp.exp1(z)
    else:
        return (np.exp(-z)-z*En(n-1,z))/(n-1)

def psi(r,y):
    def integral(integrando, r, y):  
        return quad(integrando, 0, a, args=(r, y))[0]         
    
    def L(d,y,a):
        return -np.imag(En(1,(d+1j*y)*a)) + np.real(En(2,(d+1j*y)*a))/a

    # para r < c
    def integrando1(nu, r, y):
        return (nu*np.sin(nu*y) + np.cos(nu*y))*sp.iv(1,nu*r)*sp.kv(0,nu*c)*nu/(nu**2+1)
    
    def psi1(r,y): 
        M = lambda r,y: r*np.exp(-y)*sp.jv(1,r)*sp.hankel1(0,c)
        return -4*np.pi**2*1j*c*M(r,y) - 8*c*r * integral(integrando1,r,y) + L(np.abs(r-c),y,a)/(2*np.sqrt(r*c))

    # para r > c
    def integrando2(nu, r, y):
        return (nu*np.sin(nu*y) + np.cos(nu*y))*sp.iv(0,nu*c)*sp.kv(1,nu*r)*nu/(nu**2+1)
    
    def psi2(r,y): 
        M = lambda r,y: r*np.exp(-y)*sp.jv(0,c)*sp.hankel1(1,r)
        return -4*np.pi**2*1j*c*M(r,y) + 8*c*r * integral(integrando2,r,y) + L(np.abs(r-c),y,a)/(2*np.sqrt(r*c))

    # Función partida en 2D
    sol = np.zeros(np.shape(r))
    for i in range(len(r[:,0])):
        for j in range(len(r[0,:])):
            sol[i,j] = psi1(r[i,j],y[i,j])[0] if (r[i,j]<c) else psi2(r[i,j],y[i,j])[0]
    return sol

# Parámetros
c = sp.jn_zeros(0,1)
a = 20

# Variables
r = np.linspace(0, 10, 50)
y = np.linspace(0, 5, 50)
R, Y = np.meshgrid(r, y)

# Plot
cs = plt.contour(R, Y, np.real(psi(R,Y)), levels = [4])
plt.axhline(0, color = "black")
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()

plt.show()

# # Como extraer contours con matplotlib:
# p = cs.collections[0].get_paths()[0]
# v = p.vertices
# x = v[:,0]
# y = v[:,1]

# plt.plot(x, y)
# plt.show()