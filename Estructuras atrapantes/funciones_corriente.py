import scipy.special as sp
from scipy.integrate import quad
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def En(n, z): 
    """
    Generalized exponential integral E_n for complex arguments.
    """
    if n <= 0:
        raise Exception("n must be greater than 0.")
    elif n == 1:
        return sp.exp1(z)
    else:
        return (np.exp(-z)-z*En(n-1,z))/(n-1)
    
def find_roots(f, x0s):
    f0s = f(x0s)
    f0_is_positive = (f0s>0)
    sign_changes = f0_is_positive[:-1] != f0_is_positive[1:]
    return fsolve(f, x0s[:-1][sign_changes])

    
class McIver1997:
    def __init__(self, R=sp.jn_zeros(0,1)[0], a=30):
        self.R = R
        self.a = a

    def phi(self, r, y):
        a = self.a
        R = self.R
     
        def N(l, y, a):
            return np.real(En(1,(l+1j*y)*a)) + np.imag(En(2, (l+1j*y)*a))/a

        def integrando(nu, r, y, R):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            return (nu*np.cos(nu*y)-np.sin(nu*y))*sp.iv(0,nu*rminus)*sp.kv(0,nu*rplus)*nu/(nu**2+1)

        def integral(r, y, R):
            return quad(integrando, 0, a, args=(r, y, R))[0]        

        vectorized_integral = np.vectorize(integral)

        def phi(r, y):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            M = R*np.exp(-y)*sp.jv(0,rminus)*sp.hankel1(0,rplus)
            return 4j*(np.pi**2)*M + + 8*R*(vectorized_integral(r, y, R) + N(np.abs(r-R),y,a)/(2*np.sqrt(r*R)))
        
        return phi(r, y)

        
    def psi(self, r, y):
        a = self.a
        R = self.R

        def L(l, y, a):
            return -np.imag(En(1,(l+1j*y)*a)) + np.real(En(2, (l+1j*y)*a))/a

        def integrando(nu, r, y, R):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            return (nu*np.sin(nu*y)+np.cos(nu*y))*sp.iv(r<R,nu*rminus)*sp.kv(r>R,nu*rplus)*nu/(nu**2+1)

        def integral(r, y, R):
            return quad(integrando, 0, a, args=(r, y, R))[0]        

        vectorized_integral = np.vectorize(integral)

        def psi(r, y):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            M = r*R*np.exp(-y)*sp.jv(r<R,rminus)*sp.hankel1(r>R,rplus)
            return -4j*(np.pi**2)*M + ((-1)**(r<R))*8*r*R*(vectorized_integral(r, y, R) + L(np.abs(r-R),y,a)/(2*np.sqrt(r*R)))
        
        return psi(r, y)

class Hulme1983:
    def __init__(self, R=sp.jn_zeros(0,1)[0], d=2, N=400):
        self.R, self.d, self.N = R, d, N

        self.k0 = fsolve(lambda x: x*np.tanh(x*d)-1, 1, xtol=1e-8)[0]

        C0s = np.linspace(0,1,N*5)*N*np.pi/d
        C0s = C0s + C0s[1]
        self.coefs = find_roots(lambda x: np.sin(d*x) + np.cos(d*x)/x, C0s)

    def psi(self, r, y): 
        R, d, N = self.R, self.d, self.N
        k0, C = self.k0, self.coefs
    
        def A0(y):
            return 4*(k0**2-1)/(d*(k0**2-1)+1)*np.sinh(k0*(d-y))*np.cosh(k0*d)
        
        def An(n, y):
            return 4*(C[n]**2+1)/(d*(C[n]**2+1)-1)*np.sin(C[n]*(d-y))*np.cos(C[n]*d)

        def series_term_generator(r, y):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            for n in range(N):
                term = An(n,y)/(2*C[n]*np.sqrt(r*R))*np.exp(-C[n]*(np.abs(r-R))) 
                if C[n] < 200:
                    term = An(n,y)*sp.iv(r<R, C[n]*rminus)*sp.kv(r>R, C[n]*rplus)
                yield term

        def psi(r, y):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)

            series = sum(series_term_generator(r,y))
            psi1 = -1j*(np.pi**2)*(R*r)*A0(y)*sp.jv(r<R, k0*rminus)*sp.hankel1(r>R, k0*rplus)
            psi2 = (2*np.pi)*(R*r)*series*(-1)**(r>R)

            return psi1 + psi2
        
        return psi(r, y)

    def phi(self, r, y):
        R, d, N = self.R, self.d, self.N
        k0, C = self.k0, self.coefs

        def A0(y):
            return 4*(k0**2-1)/(d*(k0**2-1)+1)*np.cosh(k0*(d-y))*np.cosh(k0*(d))

        def An(n, y):
            return 4*(C[n]**2+1)/(d*(C[n]**2+1)-1)*np.cos(C[n]*(d-y))*np.cos(C[n]*(d))
        
        def series_term_generator(r, y):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            for n in range(N):
                term = An(n,y)/(2*C[n]*np.sqrt(r*R))*np.exp(-C[n]*(np.abs(r-R))) 
                if C[n] < 200:
                    term = An(n,y)*sp.iv(0, C[n]*rminus)*sp.kv(0, C[n]*rplus)
                yield term
        
        def phi(r, y):
            rplus = r*(r>R) + R*(r<R)
            rminus = r*(r<R) + R*(r>R)
            series = sum(series_term_generator(r,y))
            return A0(y)*np.pi**2*R*1j*sp.jv(0, k0*rminus)*sp.hankel1(0, k0*rplus)+2*np.pi*R*series

        return phi(r, y)

if __name__ == "__main__":
    # Parámetros
    R = sp.jn_zeros(0,2)[0]
    a = 100
    psi1 = McIver1997(R=R, a=a).psi
    psi2 = Hulme1983(R=R, N=1000, d=15).psi

    # Variables
    r = np.linspace(0.01, 10, 100)
    y = np.linspace(0.01, 5, 100)

    # Plot
    plt.plot(r, np.abs(psi1(r,0)), label="McIver")
    plt.plot(r, np.real(psi2(r,0)), label="Hulme")
    plt.legend()

    plt.show()
    R, Y = np.meshgrid(r, y)

    # Plot
    cs = plt.contour(R, Y, np.abs(psi2(R,Y)), cmap = "PuOr_r")
    cs = plt.contourf(R, Y, np.angle(psi2(R,Y)))

    plt.axhline(0, color = "black")
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()