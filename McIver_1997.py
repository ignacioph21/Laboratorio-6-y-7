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

def psi_escalar(r,y, c=sp.jn_zeros(0,1)[0], a=30):    
    def L(d,y,a):
        return (-np.imag(En(1,(d+1j*y)*a)) + np.real(En(2,(d+1j*y)*a))/a)
    
    def integral(integrando, r, y):  
        return quad(integrando, 0, a, args=(r,y), complex_func=True)[0]

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
    return psi1(r,y) if (r<c) else psi2(r,y)

def psi(r,y):
    sol = np.zeros(np.shape(r))
    for i in range(len(r[:,0])):
        for j in range(len(r[0,:])):
            sol[i,j] = psi_escalar(r[i,j],y[i,j])
    return sol

if __name__ == "__main__":
    # Parámetros
    c = sp.jn_zeros(0,1)
    a = 20

    # Variables
    r = np.linspace(0, 10, 100)
    y = np.linspace(0, 5, 100)
    R, Y = np.meshgrid(r, y)

    # Plot
    cs = plt.contour(R, Y, np.real(psi(R,Y)), levels = [4])
    plt.axhline(0, color = "black")
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
