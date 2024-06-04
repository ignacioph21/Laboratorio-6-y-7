import numpy as np
from scipy.sparse import csc_array, dok_matrix, dia_array
from scipy.sparse.linalg import lsqr, minres

def D(N, ds=1):
    """
    Operador de segundo orden para calcular la primera derivada.
    """
    D = -np.eye(N, k=-1) + np.eye(N, k=1)
    D[ 0, 0:3] = np.array([-3, 4,-1])
    D[-1,-3: ] = np.array([ 1,-4, 3])
    return D/(2*ds)

def gradient_operator(N, ds=1):
    """
    Operador gradiente para una función de 2 dimensiones definido segun Moisy (2009).
    """
    M = N**2
    G = np.zeros((2*M,M))
    nabla = D(N, ds)
    for i in range(N):        
        G[N*i:N*(i+1), N*i:N*(i+1)] = nabla
        G[M+i::N, i::N] = nabla
    return G

def sparse_gradient_operator(N, ds=1):
    """
    Operador gradiente para una función de 2 dimensiones (matriz cuadrada de NxN) definido segun Moisy (2009) en su versión _scipy sparse_ por el bien de la computadora. 
    """
    M = N**2
    # Gradient operator
    G = dok_matrix((2*M,M))
    nabla = D(N, ds)
    for i in range(N):        
        G[N*i:N*(i+1), N*i:N*(i+1)] = nabla
        G[M+i::N, i::N] = nabla
    G = csc_array(G)
    G.eliminate_zeros()

    return G

def gradient(f, ds=1):
    if f.shape[0] != f.shape[1]:
        raise Exception("gradient: f must be square")
    
    N = f.shape[0]
    G = sparse_gradient_operator(N, ds)

    df = G.dot(np.reshape(f, N**2))
    fx = np.reshape(df[:N**2], (N,N))
    fy = np.reshape(df[N**2:], (N,N))

    return fx, fy

def lsqrinvgrad(fx, fy, cal=1):
    if fx.shape != fy.shape:
        raise Exception("lsqrinvgrad: fx and fy must have the same shape.")
    if fx.shape[0] != fx.shape[1]:
        raise Exception("lsqrinvgrad: fx and fy must be square")
    
    N = fx.shape[0]
    M = N**2

    df = np.zeros(2*M)
    df[:M] = np.reshape(fx, M)
    df[M:] = np.reshape(fy, M)

    G = sparse_gradient_operator(N, cal)
    f = lsqr(G, df)[0]
    return np.reshape(f, (N,N))
