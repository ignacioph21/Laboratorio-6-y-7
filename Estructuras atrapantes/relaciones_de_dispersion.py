import numpy as np
from scipy.optimize import fsolve

# Parámetros
sigma = 0.07275  # N/m
h     = 5e-2     # m
rho   = 1000     # Kg / m^3
g     = 9.8      # m/s^2
nu    = 1e-6     # m^2 s

# Relaciones de dispersión
def w_gravito_capilares(k):
    return np.sqrt(k*(g + sigma*k**2/rho)*np.tanh(k*h))

def w_gravedad(k):
    return np.sqrt(k*g*np.tanh(k*h))

def w_capilares(k):
    return np.sqrt(k*(sigma*k**2/rho)*np.tanh(k*h))

def k_gravito_capilares(ws):
    return np.array([*map(lambda w_: fsolve(lambda k_: w_gravito_capilares(k_)-w_, 0)[0], ws)])

def k_gravedad(ws):
    return np.array([*map(lambda w_: fsolve(lambda k_: w_gravedad(k_)-w_, 0)[0], ws)])

def k_capilares(ws):
    return np.array([*map(lambda w_: fsolve(lambda k_: w_capilares(k_)-w_, 0)[0], ws)])

k_rel_disp_dict = {
    "Gravito-Capilares": k_gravito_capilares,
    "Gravedad": k_gravedad,
    "Capilares": k_capilares,
}

w_rel_disp_dict = {
    "Gravito-Capilares": w_gravito_capilares,
    "Gravedad": w_gravedad,
    "Capilares": w_capilares,
}

# Definición del factor de adimensionalización
def K(w): 
    return w**2/g

def w(K):
    return np.sqrt(K*g)