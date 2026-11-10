import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-50, 50, 2000)
y = np.linspace(-50, 50, 2000)
X, Y = np.meshgrid(x, y)
Z = np.cbrt(114 - X**3 - Y**3)   # superfície real para n = 114

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Superfície de $x^3+y^3+z^3=114$ (sem soluções inteiras conhecidas)')
plt.show()