import matplotlib.pyplot as plt
import numpy as np

def extrude(x, y, d):
    # x, y must be the same size. 
    for i in range(1, len(x)-1):
        point_backward = np.array([x[i-1], y[i-1]])
        point_actual = np.array([x[i], y[i]])
        point_forward = np.array([x[i+1], y[i+1]])

        vector_forward = point_actual - point_backward 
        vector_backward = point_forward - point_actual

        normal_forward = np.array([vector_forward[1], -vector_forward[0]])
        normal_backward = np.array([vector_backward[1], -vector_backward[0]])

        normal = (normal_forward + normal_backward)/2
        normal /= np.sqrt(sum(normal**2)) 
        
        new_point = point_actual + d * normal

        x = np.append(x, new_point[0])
        y = np.append(y, new_point[1]) 
    return [x, y]
        
t = np.linspace(0, 2*np.pi, 1000)
x = np.cos(t)
y = np.sin(t)
d = -0.5

extruded = extrude(x, y, d)
print(len(x), len(extruded[0]))
plt.plot(x, y)
plt.plot(extruded[0], extruded[1])
plt.show()
