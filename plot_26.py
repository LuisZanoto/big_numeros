import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = np.cbrt(26 - X**3 - Y**3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# superfície real
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0)

# pontos inteiros que satisfazem a equação
pts = []
for a in range(-20, 21):
    for b in range(-20, 21):
        for c in range(-20, 21):
            if a**3 + b**3 + c**3 == 26:
                pts.append((a, b, c))

pts = np.array(pts)
if len(pts) > 0:
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='red', s=50)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()