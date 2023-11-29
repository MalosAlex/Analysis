from math import exp, pi, sqrt


def f(x):
    return x * x


def gpd(inf, nr):
    inc = (inf * 2) / nr
    total = 0
    x = -1 * inf
    for i in range(1, nr + 1):
        x = x + inc
        total += inc * exp(-1 * f(x))
    return total


a = int(input("Chose an a: "))
n = int(input("Chose how many trapeziums to use: "))
print("The definite integral from " + str(-1 * a) + " to " + str(a) + " of e^(-x^2) is: " + str(gpd(a, n)))
print("The square root of pi is: " + str(sqrt(pi)))
