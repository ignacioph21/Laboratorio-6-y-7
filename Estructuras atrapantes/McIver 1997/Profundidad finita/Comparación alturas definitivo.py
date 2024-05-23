from scipy.special import jv, hankel1, iv, kv,jn_zeros, exp1
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

c = jn_zeros(0,1)
C =  16 
N = 390 


def A_int0(y, d, k0):
    return 4*(k0**2-1**2)/(d*(k0**2-1**2)+1)*np.sinh(k0*(d-y))*np.cosh(k0*(d))

def A_intn(y, d, cn):
    return 4*(cn**2+1**2)/(d*(cn**2+1**2)-1)*np.sin(cn*(d-y))*np.cos(cn*(d))

def term(cn, rplus, rminus, idxplus, idxminus, y, d):
    A = A_intn(y, d, cn)
    if cn < 200:  
        B = iv(idxminus, cn*rminus)
        C = kv(idxplus, cn*rplus)
        return A*B*C
    D = 1/(2*cn*np.sqrt(rplus*rminus)) * np.exp(-cn*(abs(rplus-rminus))) 
    return A*D 


def psi(r, y, d):
    # Encontrar el k0.
    k0 = fsolve(lambda x: x*np.sinh(x*d)-np.cosh(x*d), 1, xtol=1e-14)
    print(k0*np.sinh(k0*d)-1*np.cosh(k0*d))

    # Encontrar los c_n.
    idxs = np.arange(1, N+1, 1)
    intervals = (2*idxs-1)*(np.pi/(2*d)+1e-5)
     
    cn = fsolve(lambda x: np.tan(x*d)+1/x, intervals) 
    print(sum(np.tan(cn*d)+1/cn))

    
    rplus = r*(r>c) + c*(r<c)
    rminus = r*(r<c) + c*(r>c)
    idxminus = r<c
    idxplus = r>c
    terms = [term(cn[n], rplus, rminus, idxplus, idxminus, y, d) for n in range(0, N-1)]
    return -A_int0(y, d, k0)*np.pi**2*c*r*1j*jv(idxminus, k0*rminus)*hankel1(idxplus, k0*rplus)-(-1)**idxminus*2*np.pi*c*r*sum(terms)

rs = np.linspace(0.05, 3.10, 50) 
ys = np.linspace(0, 2, 50)  
R, Y = np.meshgrid(rs, ys)

ds = np.arange(2, 15, 1)
xs = []
ys = []
ds_prima = []
for d in ds:
    cs = plt.contour(R, Y, np.real(psi(R, Y, d)), levels = [C][::-1]) 
    if len(cs.collections[0].get_paths()):
        p = cs.collections[0].get_paths()[0] # TODO: Concatenar todos.
        v = p.vertices
        x = v[:,0]
        y = v[:,1]
        xs.append(x)
        ys.append(y)
        ds_prima.append(d)

a = 100 
def En(n, z): 
    if n <= 0:
        raise Exception("n must be greater than 0.")
    elif n == 1:
        return exp1(z)
    else:
        return (np.exp(-z)-z*En(n-1,z))/(n-1)

def L(d,y,a):
    return -np.imag(En(1,(d+1j*y)*a)) + np.real(En(2, (d+1j*y)*a))/a

def integrando(nu, rplus, rminus, idxplus, idxminus, y):
    return (nu*np.sin(nu*y) + np.cos(nu*y))*iv(idxminus,nu*rminus)*kv(idxplus,nu*rplus)*nu/(nu**2+1)

def integral(rplus, rminus, idxplus, idxminus, y):  
    return quad(integrando, 0, a, args=(rplus, rminus, idxplus, idxminus, y))[0]        

vectorized_integral = np.vectorize(integral)

def psi_infinite(r,y):
    rplus = r*(r>c) + c*(r<c)
    rminus = r*(r<c) + c*(r>c)
    idxplus = r>c
    idxminus = r<c
    M = r*np.exp(-y)*jv(idxminus,rminus)*hankel1(idxplus,rplus)
    return -4*np.pi**2*1j*c*M + (-1)**idxminus * 8*c*r * vectorized_integral(rplus, rminus, idxplus, idxminus,y)  + L(np.abs(r-c),y,a) / (2*np.sqrt(r*c))

cs = plt.contour(R, Y, np.real(psi_infinite(R, Y)), levels = [C][::-1])   
if len(cs.collections[0].get_paths()):
    p = cs.collections[0].get_paths()[0] # TODO: Concatenar todos.
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
    xs.append(x)
    ys.append(y)
    ds_prima.append("inf")




plt.close()

for i in range(len(ds_prima)): 
    plt.plot(xs[i], ys[i], label=f"d = {ds_prima[i]}") 
    #     plt.plot(np.convolve(xs[i], [1/2,1/2])[:-2], np.convolve(ys[i], [1/2,1/2])[:-2], label=f"d = {ds_prima[i]}") 


plt.legend()    


plt.gca().invert_yaxis()
plt.xlabel("r")
plt.ylabel("y") 
plt.show()
