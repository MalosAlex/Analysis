import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[8, 0], [0, -3]])


# The function
def f(x, A):
    return 0.5 * np.dot(x.T, np.dot(A, x))


# The gradient
def grad_f(x, A):
    return np.dot(A, x)


# Creating the mesh
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Calculating the function value for each point in the mesh
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = f(np.array([x[i], y[j]]), A)

# Plotting the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Plotting 3 contour lines
contour_levels = np.linspace(np.min(Z), np.max(Z), 10)
ax.contour(X, Y, Z, contour_levels, offset=np.min(Z), cmap='viridis')

# Plotting the gradient at 3 points
points = np.array([[2, 2], [-2, -2], [2, -2]])
for point in points:
    ax.quiver(point[0], point[1], f(point, A), grad_f(point, A)[0], grad_f(point, A)[1], 0, color='red')

# Setting axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Contour plot of the function f(x) = 0.5 * x.T * A * x')

# Showing the plot
plt.show()