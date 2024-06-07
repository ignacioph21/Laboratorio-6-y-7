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

def L(d,y,a):
    return -np.imag(En(1,(d+1j*y)*a)) + np.real(En(2, (d+1j*y)*a))/a

def integrando(nu, rplus, rminus, idxplus, idxminus, y):
    return (nu*np.sin(nu*y) + np.cos(nu*y))*sp.iv(idxminus,nu*rminus)*sp.kv(idxplus,nu*rplus)*nu/(nu**2+1)

def integral(rplus, rminus, idxplus, idxminus, y):  
    return quad(integrando, 0, a, args=(rplus, rminus, idxplus, idxminus, y))[0]        

vectorized_integral = np.vectorize(integral)

def psi(r,y):
    rplus = r*(r>c) + c*(r<c)
    rminus = r*(r<c) + c*(r>c)
    idxplus = r>c
    idxminus = r<c
    M = r*np.exp(-y)*sp.jv(idxminus,rminus)*sp.hankel1(idxplus,rplus)
    return -4*np.pi**2*1j*c*M + (-1)**idxminus * 8*c*r * ( vectorized_integral(rplus, rminus, idxplus, idxminus,y)  + L(np.abs(r-c),y,a) / (2*np.sqrt(r*c)) ) # TODO: acá agregué un paréntesis, revisar

c = sp.jn_zeros(0,1)
a = 25

r = np.linspace(0, 10, 100)
y = np.linspace(0, 5, 100)

R, Y = np.meshgrid(r,y)

plt.gca().set_aspect('equal')
cs = plt.contour(R, Y, np.real(psi(R,Y)), levels = [20, 16, 12, 8, 4][::-1])
plt.hlines(0, 0, 10)
plt.scatter([c], [0])
plt.xlabel("r")
plt.ylabel("y")
plt.gca().invert_yaxis()


plt.show()
