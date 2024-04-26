from scipy.special import jv, hankel1, iv, kv,jn_zeros
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

d = 5 # Esto sería d=Kh=w^2 h/g. 

c = jn_zeros(0,1)

N = 20 

# Encontrar el k0.
k0 = fsolve(lambda x: x*np.sinh(x*d)-np.cosh(x*d), 1, xtol=1e-14)
print(k0*np.sinh(k0*d)-1*np.cosh(k0*d))

# Encontrar los c_n.
idxs = np.arange(1, N+1, 1)
intervals = (2*idxs-1)*(np.pi/(2*d)+1e-5)

cn = fsolve(lambda x: np.tan(x*d)+1/x, intervals)  
print(sum(np.tan(cn*d)+1/cn))

# Calcular los A_n.
def A0(y):
    return 4*(k0**2-1**2)/(d*(k0**2-1**2)+1)*np.cosh(k0*(d-y))*np.cosh(k0*(d))

def An(y, n):
    return 4*(cn[n]**2+1**2)/(d*(cn[n]**2+1**2)-1)*np.cos(cn[n]*(d-y))*np.cos(cn[n]*(d))

def Rm(r,y):
    rplus = r*(r>c) + c*(r<c)
    rminus = r*(r<c) + c*(r>c)
    terms = [An(y, n)*iv(0, cn[n]*rminus)*kv(0, cn[n]*rplus) for n in range(0, N-1)]
    return A0(y)*np.pi**2*c*1j*jv(0, k0*rminus)*hankel1(0, k0*rplus)+2*np.pi*c*sum(terms)

rs = np.linspace(0, 10, 100)
ys = np.linspace(0, d, 100)
R, Y = np.meshgrid(rs, ys)

grad = np.gradient(np.real(Rm(R,Y)), rs, ys)

ux = grad[1]
uy = grad[0]

plt.streamplot(R, Y, ux, uy, density=2)


# plt.contour(R, Y, np.real(Rm(R,Y)), 15)
plt.gca().invert_yaxis()


plt.show()
