"""
Idea:
1. Encontrar una raíz de psi - h = 0
2. Encontrar derivadas parciales de psi en ese punto.
3. Moverse ortogonal a las derivadas parciales (la curva implícita tiene pendiente -fr/fy o algo así). 
4. Volver a buscar una raíz de la función (si existe), con el parámetro p0 dado por (3).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, minimize_scalar
from scipy.special import jn_zeros
from McIver_1997 import psi_escalar

class NoRootsFoundError(Exception):
    pass

def normalize(X):
    """Para obtener la dirección de un vector o signo de un escalar"""
    return X / np.linalg.norm(X)

def rk4_step(dX, X, ds):
    """Devuelve el paso de la curva con runge-kutta 4. No toma variable independiente en el argumento."""
    K1 = dX(X)
    K2 = dX(X + K1*ds/2)
    K3 = dX(X + K2*ds/2)
    K4 = dX(X + K3*ds)
    K = K1 + 2*K2 + 2*K3 + K4
    return normalize(K)*ds

def step_sign(X, step, center_point, orientation):
    """Devuelve el signo que debería tomar un paso si se quiere recorrer la curva con cierta orientación"""
    direction = np.array([-(X[1] - center_point[1]), X[0] - center_point[0]])
    match orientation: 
        case "cc":
            sign = normalize(step @ direction)
        case "cw":
            sign = - normalize(step @ direction)
    return sign

def get_root_of_func2d_slice(f, var_index, fixed_arg_value, bounds, tol=1e-2):
    """Para obtener la raíz de una curva de la función 1D dada por una función 2D con un argumento fijo. Usa el método de bisección de scipy."""
    match var_index:
        case 0:
            f_ = lambda x_: f(x_, fixed_arg_value)
        case 1:
            f_ = lambda y_: f(fixed_arg_value, y_)
        case _:
            raise Exception("")

    min_f = minimize_scalar(f_, bounds=bounds)
    max_f = minimize_scalar(lambda x: -f_(x), bounds=bounds)

    if min_f.fun*(-max_f.fun) > 0: 
        raise NoRootsFoundError("get_root_of_func2d_slice: no roots found within bounds.")
    
    root = np.ones(2)*fixed_arg_value
    root[var_index] = bisect(f_, min_f.x, max_f.x, xtol=tol)

    return root

def get_first_root(f, bounds, dx, tol=1e-2):
    """
    Busca una raíz de la función de la siguiente forma:
        1. Fija la primera variable, y busca una raíz de la función 1D dada por la variable fija. 
        2. Si encuentra una raíz, la devuelve. Si no, suma dx a la primera variable. 
        3. Repite el proceso. 
    """
    x_bounds, y_bounds = bounds
    xi, xf = x_bounds
    xs = np.arange(xi, xf, dx)

    # Buscar un valor de r donde el corte cambia de signo al variar y
    root = []
    for x in xs:
        try: 
            root = get_root_of_func2d_slice(f, var_index=1, fixed_arg_value=x, bounds=y_bounds, tol=tol)
            break
        except NoRootsFoundError:
            pass
    
    if len(root) == 0:
        raise NoRootsFoundError("get_first_root: No roots found within bounds.")
    
    return root

def func2d_root_search(f, bounds, ds=1e-2, ss =1e-1, orientation="cc", center_point=np.array([0,0]), tol=1e-4, maxiter=1000):
    """
    Busca la curva de una función 2D dada por f(x,y) == 0. 
        Bounds: (e.g. [(xi, xf), (yi, yf)]) determina el dominio donde busca la curva. 
        ss: (search-size) el tamaño de la ventanita que determina los bounds donde se va a buscar la próxima raíz. 
            Si es muy chico, capaz "se escapa la raíz". Si es muy grande, y la función no es biyectiva, puede encontrar otra raíz primero y "romper la curva". 
        orientation, center_point: sirven para dererminar en que sentido se recorre la curva. 
    """
    x_bounds, y_bounds = bounds
    xi, xf = x_bounds
    yi, yf = y_bounds
    
    def dX(X):
        r, y = X
        f0 = f(r, y)
        fr = f(r+ds, y) - f0
        fy = f(r, y+ds) - f0
        dX = np.array([fy, -fr])
        return normalize(dX)

    X = get_first_root(f, bounds, dx=ds, tol=tol)
    yield X

    s = 0
    i = 0
    x, y = X
    while (xi <= x <= xf and yi <= y <= yf) and i < maxiter:
        step = rk4_step(dX, X, ds)
        X = X + step*step_sign(X, step, center_point, orientation)
        x, y = X
        dx, dy = step

        try:
            if abs(dx/dy) <= 1:  # poco cambio en r, mucho cambio en y --> congelo y, busco raíz en r 
                X = get_root_of_func2d_slice(f, var_index=0, fixed_arg_value = y, bounds = (x-ss, x+ss), tol=tol)
            else:                # poco cambio en y, mucho cambio en r --> congelo r, busco raíz en y
                X = get_root_of_func2d_slice(f, var_index=1, fixed_arg_value = x, bounds = (y-ss, y+ss), tol=tol)

        except Exception as e:
            print(e)
            break

        i += 1
        yield X

h = 12
c = 2.4048
ds = 2e-2

r_bounds = (0.1, c-ds)
y_bounds = (-0.1, 10)
bounds = (r_bounds, y_bounds)

f = lambda x, y: np.real(psi_escalar(x,y) - h)

roots = func2d_root_search(f, bounds, ds=ds, ss=1e-1, orientation="cw", center_point=np.array([c,0]))
curva = np.array(list(roots))

plt.plot(curva[:,0], curva[:,1], ".-")

r_bounds = (c+ds, 10)
bounds = (r_bounds, y_bounds)
roots = func2d_root_search(f, bounds, ds=ds, ss=1e-1, orientation="cw", center_point=np.array([c,0]))
curva = np.array(list(roots))

plt.plot(curva[:,0], curva[:,1], ".-")

plt.axhline("0")
plt.gca().set_aspect("equal")
plt.show()