# I selected the random f(x) = x^2 + 8x + 18, which is convex
# The nonconvex f chosen is f(x) = - x ^3

import matplotlib.pyplot as plt
import numpy as np


def f_conv(x):
    return x ** 2 + 8 * x + 18


def f_conv_der(x):
    return 2 * x + 8


def f_non_conv(x):
    fun = 2 * x ** 3 - 8 * x
    return fun


def f_non_der(x):
    return 6 * x ** 2 - 8


# for conv
# def grad_des_met_conv(xn, nl):
#     rez = xn - nl * f_conv_der(xn)
#     return rez
#

def grad_des_met_non(xn, nl):
    rez = xn - nl * f_non_der(xn)
    return rez


a = int(input("x0 = "))
xs = [a]
n = int(input("Number of operations: "))
lr = float(input("Learning rate: "))
# (a)
for i in range(n):
    xi = grad_des_met_non(xs[i], lr)
    xs.append(xi)

# Creating a list of iterations
iterations = list(range(n+1))

plt.plot(iterations, xs, c="red", marker='x', label="xn")
plt.scatter(range(n + 1), xs, c="red", marker='x', label='xn')
plt.xlabel('Iteration')
plt.ylabel('xn')
plt.legend()
plt.title("Gradient descent method descent")
plt.show()
