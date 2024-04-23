"""
Idea:
1. Encontrar una raíz de psi - h = 0
2. Encontrar derivadas parciales de psi en ese punto.
3. Moverse ortogonal a las derivadas parciales (la curva implícita tiene pendiente -fr/fy o algo así). 
4. Volver a buscar una raíz de la función (si existe), con el parámetro p0 dado por (3).
PARA HACER: 
- Agregar un P0 en func2d_root_search
- Agregar opción singularities para que esquive ciertos puntos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, minimize_scalar
from scipy.special import jn_zeros
from fuciones_corriente import McIver1997

class NoRootsFoundError(Exception):
    pass

def normalize(X):
    """Devuelve el signo/dirección del float/ndarray X"""
    return X / np.linalg.norm(X)

def root_of_2Dfunc_trace(f, var_index, bounds, fixed_arg_value, tol=1e-4):
    """
    Encuentra una raíz de una función 2D fijando uno de sus argumentos y variando el otro dentro de los `bounds` dados. 
    Usa la función `bisect` de Scipy.
    ## Parámetros
    f : function
        Tunción 2D. 
    var_index : int
        0 o 1, índice de la variable "libre". 
    bounds : tuple
        Límites del intervalo donde se vá a buscar la raíz. 
    fixed_arg_value : float
        Valor de la variable fija. 
    tol : float
        Tolerancia del método de bisección al computar la raíz. 
    # Devuelve
    root : ndarray
        Coordenadas de la raíz encontrada. 
    """
    match var_index:
        case 0:
            f_ = lambda x_: f(x_, fixed_arg_value)
        case 1:
            f_ = lambda y_: f(fixed_arg_value, y_)
        case _:
            raise Exception("2D Function: var_index must be 0 or 1.")

    min_f = minimize_scalar(f_, bounds=bounds)
    max_f = minimize_scalar(lambda x: -f_(x), bounds=bounds)

    if min_f.fun*(-max_f.fun) > 0: 
        raise NoRootsFoundError("root_of_func2D_trace: no roots found within bounds.")

    root = np.ones(2)*fixed_arg_value
    root[var_index] = bisect(f_, min_f.x, max_f.x, xtol=tol)

    return root

def first_root_sweep(f, bounds, dx, tol=1e-4):
    """
    Barrer el espacio para encontrar una raíz. Al final no se usa esta función.
    """
    x_bounds, y_bounds = bounds
    xi, xf = x_bounds
    xs = np.arange(xi, xf+dx, dx)

    # Buscar un valor de r donde el corte cambia de signo al variar y
    root = []
    for x in xs:
        try: 
            root = root_of_2Dfunc_trace(f, var_index=1, fixed_arg_value=x, bounds=y_bounds, tol=tol)
            break
        except NoRootsFoundError:
            pass
    
    if len(root) == 0:
        raise NoRootsFoundError("first_root_sweep: No roots found within bounds.")
    
    if dx > tol:
        x_bounds = max(xi, root[0]-dx), root[0]
        root = first_root_sweep(f, (x_bounds, y_bounds), dx=dx/10, tol=tol)
    
    return root

def implicit_2Dcurve(f, X0, 
                      bounds=[(-np.inf, np.inf),(-np.inf, np.inf)], 
                      ds=5e-2, search_size=1e-1, 
                      direction_guide=lambda X, dX: 1, 
                      tol=1e-4, maxiter=1000):
    """
    Genera puntos en una curva implícita dada por \[f(x,y)=0\]. Funciona aproximando la tangente de la curva
    en un punto usando diferencias finitas y moviéndose un paso ds a lo largo de la tangente para acercarse al siguiente punto. 
    Después de esto utiliza la función de bisección de scipy para devolver el siguiente punto dentro de la tolerancia dada.

    ## Parámetros:
    f : function
    X0 : ndarray
        Punto de partida. Debe ser (dentro de la tolerancia de) una raíz de \[f(x,y)=0\].
    bounds : list
        Lista que contiene los límites de la primera variable (tuple) y los límites de la segunda variable (tuple).
        La curva se calculará dentro de esos límites.
    ds: float
        Tamaño de los pasos sobre las tangentes.
    search_size : float
        Determina el intervalo donde la función de bisección de scipy busca el siguiente punto de la curva.
        Demasiado grande y se puede perder la continuidad.
    direction_guide : function
    Una función f(X, dX) opcional para forzar que en X el paso dX se tome en una cierta dirección (impone un signo).
    tol: float
        Tolerancia del método a errores.
    maxiter: int
        Número máximo de pasos dados a lo largo de la curva.

    ## Genera
        X : ndarray
        Las coordenadas del siguiente punto de la curva.
    """
    def dX(X):
        r, y = X
        f0 = f(r, y)
        fr = f(r+ds, y) - f0
        fy = f(r, y+ds) - f0
        dX = np.array([fy, -fr])
        return normalize(dX)*ds
    
    x_bounds, y_bounds = bounds
    xi, xf = x_bounds
    yi, yf = y_bounds
    
    X = X0
    yield X

    x, y = X
    i = 0
    while (xi <= x <= xf and yi <= y <= yf) and i < maxiter:
        step = dX(X)
        X = X + direction_guide(X, step)*step

        x, y = X
        dx, dy = np.abs(step)

        try:
            if dx/dy <= 1:  # poco cambio en r, mucho cambio en y --> congelo y, busco raíz en x 
                X = root_of_2Dfunc_trace(f, var_index=0, fixed_arg_value=y, bounds=(x-search_size, x+search_size), tol=tol)
            else:           # poco cambio en y, mucho cambio en r --> congelo x, busco raíz en y
                X = root_of_2Dfunc_trace(f, var_index=1, fixed_arg_value=x, bounds=(y-search_size, y+search_size), tol=tol)

        except NoRootsFoundError:
            print(f"implicit_2d_curve: Exited. Next point on curve not found.")
            break

        except Exception:
            print(Exception)
            break
        
        yield X
        i += 1

        if i > 10 and np.linalg.norm(X-X0) < ds: 
            print(f"implicit_2d_curve: Exited. Reached start point.")
            break

def straight_guide(direction):
    def guide(X, step):
        """Devuelve 1 si el producto escalar de `step` y `direction` es mayor a 0, o -1 en caso contrario."""
        return normalize(step @ direction)
    return guide

def circular_guide(center_point, orientation):
    def guide(X, step):
        """Devuelve 1 si la orientación de `step` respecto a `center_point` corresponde a `orientation`, -1 si no"""
        direction = np.array([-(X[1] - center_point[1]), X[0] - center_point[0]])
        match orientation: 
            case "ccw":
                sign_correction = normalize(step @ direction)
            case "cw":
                sign_correction = - normalize(step @ direction)
        return sign_correction
    return guide

h = 12
c = jn_zeros(0,2)[1]
a = 50
ds = 2e-2

psi_escalar = McIver1997(c=c, a=a).psi_escalar
f = lambda x, y: np.real(psi_escalar(x,y) - h)

y_bounds = (-0.1, np.inf)
r_bounds = (ds, np.inf)
bounds = (r_bounds, y_bounds)

X0 = root_of_2Dfunc_trace(f, 0, (ds, c-ds), y_bounds[0], tol=1e-4)
roots = implicit_2Dcurve(f, X0, bounds, ds=ds, direction_guide=circular_guide((c,0),"cw"), tol=1e-4, maxiter=1000)
curva = np.array(list(roots)).T

plt.figure()
ax = plt.axes()
ax.plot(curva[0], curva[1], ".-", color="navy", lw=1.5)
ax.axhline(0, color = "k")
ax.axvline(0, color = "k")
ax.invert_yaxis()
ax.set_aspect("equal")
plt.show()