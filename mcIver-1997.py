import scipy.special as sp
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def En(n, z): 
    if n <= 0:
        raise Exception("n must be greater than 0.")
    elif n == 1:
        return sp.exp1(z)
    else:
        return (np.exp(-z)-z*En(n-1,z))/(n-1)

def E1(z):
    return En(1,z)
def E2(z):
    return En(2,z)

def L(d,y,a):
    return -np.imag(E1((d+1j*y)*a)) + np.real(E2((d+1j*y)*a))/a

c = sp.jn_zeros(0,1)
a = 25

def integrando1(nu, r, y):
    return (nu*np.sin(nu*y) + np.cos(nu*y))*sp.iv(1,nu*r)*sp.kv(0,nu*c)*nu/(nu**2+1)

def integrando2(nu, r, y):
    return (nu*np.sin(nu*y) + np.cos(nu*y))*sp.iv(0,nu*c)*sp.kv(1,nu*r)*nu/(nu**2+1)

def integral1(r, y):  
    sol = np.zeros(np.shape(r))
    for i in range(len(r[:,0])):
        for j in range(len(r[0,:])):
            #print(np.shape(quad(integrando1, 0, a, args=(r[i,j], y[i,j]))))
            sol[i,j] = quad(integrando1, 0, a, args=(r[i,j], y[i,j]))[0]
    return sol        

def integral2(r, y):
    sol = np.zeros(np.shape(r))
    for i in range(len(r[:,0])):
        for j in range(len(r[0,:])):
            sol[i,j] = quad(integrando2, 0, a, args=(r[i,j], y[i,j]))[0]
    return sol        

def psi1(r,y):
    print(np.shape(r), np.shape(y))
    M = lambda r,y: r*np.exp(-y)*sp.jv(1,r)*sp.hankel1(0,c)
    return -4*np.pi**2*1j*c*M(r,y) - 8*c*r * integral1(r,y) + L(np.abs(r-c),y,a) / (2*np.sqrt(r*c))

def psi2(r,y):
    M = lambda r,y: r*np.exp(-y)*sp.jv(0,c)*sp.hankel1(1,r)
    return -4*np.pi**2*1j*c*M(r,y) + 8*c*r * integral2(r,y) + L(np.abs(r-c),y,a) / (2*np.sqrt(r*c))

def psi(r,y):
    #return np.concatenate(psi1(r[r<c],y), psi2(r[r>=c],y))
    return psi1(r,y)*(r<c) + psi2(r,y)*(r>=c)

r = np.linspace(0, 10, 100)
y = np.linspace(-1, 5, 100)

R, Y = np.meshgrid(r,y)

plt.gca().set_aspect('equal')
plt.contour(R, Y, np.real(psi(R,Y)), levels = [20, 16, 12, 8, 4][::-1])
plt.hlines(0, 0, 10)
plt.gca().invert_yaxis()

plt.show()