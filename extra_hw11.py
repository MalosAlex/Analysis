import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1.1
def f1(x, y):
        return x ** 2 + 3 * y ** 2


def grad_f1(x, y):
    df_dx = 2 * x
    df_dy = 6 * y
    return np.array([df_dx, df_dy])


def hessian_f1(x, y):
    hessian = np.array([[2, 0], [0, 6]])
    return hessian


# 1.2
def f2(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def grad_f2(x, y):
    df_dx = -400 * x * (y - x ** 2) - 2 * (1 - x)
    df_dy = 200 * (y - x ** 2)
    return np.array([df_dx, df_dy])


def hessian_f2(x, y):
    hessian = np.array([
        [1200 * x**2 - 400 * y + 2, -400 * x],
        [-400 * x, 200]
    ])
    return hessian


def newton_method(initial_point, gradient, hessian, tolerance=1e-6, max_iterations=1000):
    x = initial_point
    x_history = [x]
    for i in range(max_iterations):
        gradient_value = gradient(x[0], x[1])
        hessian_value = hessian(x[0], x[1])
        try:
            inverse_hessian = np.linalg.inv(hessian_value)
        except np.linalg.LinAlgError:
            print("Inverse of Hessian is not defined.")
            break
        update = -np.dot(inverse_hessian, gradient_value)
        x = x + update
        x_history.append(x)
        if np.linalg.norm(update) < tolerance:
            print(f"Converged in {i + 1} iterations.")
            return np.array(x_history)
    print(f"Did not converge in {max_iterations} iterations.")
    return np.array(x_history)


def gradient_descent(initial_point, gradient, learning_rate=0.001, tolerance=1e-6, max_iterations=100000):
    x = initial_point
    x_history = [x]
    for i in range(max_iterations):
        gradient_value = gradient(x[0], x[1])
        x = x - learning_rate * gradient_value
        x_history.append(x)
        if np.linalg.norm(gradient_value) < tolerance:
            print(f"Converged in {i + 1} iterations.")
            return np.array(x_history)
    print(f"Did not converge in {max_iterations} iterations.")
    return np.array(x_history)


def plot_surface_and_steps(func, x_values, y_values, steps, method_name, ax):
    X, Y = np.meshgrid(x_values, y_values)
    Z = func(X, Y)

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=100, cstride=100)

    # Plot the optimization path
    steps = np.array(steps)
    ax.plot(steps[:, 0], steps[:, 1], func(steps[:, 0], steps[:, 1]), color='red', marker='o', linestyle='dashed')

    # Mark the end point
    ax.scatter(steps[-1, 0], steps[-1, 1], func(steps[-1, 0], steps[-1, 1]), color='black', marker='X', s=100)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Surface Plot with {method_name} (Steps: {len(steps) - 1})\nEnd Point: {steps[-1]}')

# Example usage for the first function (x^2 + 3y^2)
initial_point = np.array([-7, 8])

# Newton's method
x_history_newton = newton_method(initial_point, grad_f1, hessian_f1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_surface_and_steps(f1, np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), x_history_newton, 'Newton', ax)

plt.show()

# Gradient descent
x_history_gradient = gradient_descent(initial_point, grad_f1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_surface_and_steps(f1, np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), x_history_gradient, 'Gradient Descent', ax)

plt.show()

# Example usage for the second function (100(y-x^2)^2 + (1-x)^2)
initial_point = np.array([-1.2, 1])

# Newton's method
x_history_newton = newton_method(initial_point, grad_f2, hessian_f2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_surface_and_steps(f2, np.linspace(-2, 2, 100), np.linspace(-1, 3, 100), x_history_newton, 'Newton', ax)

plt.show()

# Gradient descent
x_history_gradient = gradient_descent(initial_point, grad_f2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_surface_and_steps(f2, np.linspace(-2, 2, 100), np.linspace(-1, 3, 100), x_history_gradient, 'Gradient Descent', ax)

plt.show()


