import matplotlib.pyplot as plt
import numpy as np

def extrude(x, y, d):
    # x, y must be the same size.
    # La otra idea es tomar backward y forward y promediar normales, como en el otro ejemplo.
    # Hay que calcular las de los bordes "a mano" en cualquier caso, no sé si haga falta.
    new_y = np.diff(x)
    new_x = -np.diff(y)
    norms = np.sqrt(new_x**2 + new_y**2)
    
    x_p = x[:-1] + d * new_x / norms
    y_p = y[:-1] + d * new_y / norms
    
    return [x_p, y_p] 

def add_walls(x, y, z, points = 100): # z es el límite, se puede cambiar para que sea un \Delta z.
    # for i in range(-1, 1): # Idea para no copiar y pegar
    wall_x1 = x[0] * np.ones(points)
    wall_y1 = np.linspace(y[0], z, points)    
    wall_x2 = x[-1] * np.ones(points)
    wall_y2 = np.linspace(y[-1], z, points)    

    return [np.concatenate([wall_x1, wall_x2]), np.concatenate([wall_y1, wall_y2])]    
        
t = np.linspace(0, -np.pi, 1000)
x = np.cos(t)
y = np.sin(t)
d = -0.50

z = 0.5
points = 100

xp, yp = extrude(x, y, d)
wallsx1, wallsy1 = add_walls(x, y, z, points=points)
wallsx2, wallsy2 = add_walls(xp, yp, z, points=points)

allx = np.concatenate([x, xp, wallsx1, wallsx2])
ally = np.concatenate([y, yp, wallsy1, wallsy2])
plt.scatter(allx, ally)

plt.show()
