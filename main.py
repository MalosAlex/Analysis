from tabulate import tabulate
import math

results = []
sign = 1
result = float(0)
n_input = int(input("Please give an n for which to calculate the sums (n, n*2, n*4, n*8):\n>"))
p = int(input("Please give a number p, to make p additions in the sum before the subtractions:\n>"))
q = int(input("Please give a number q, to make q subtractions after the additions:\n>"))

for j in range(0, 4):
    n = n_input * (2 ** j)
    sign = 1
    result = float(0)
    for i in range(1, n):
        result = result + (1/i * sign)
        sign = sign * -1
    results.append(result)

data = [
    ["n", "p", "q", "result", "e^result = ln x"],
    [n_input, 1, 1, results[0], math.exp(results[0])],
    [n_input * 2, 1, 1, results[1], math.exp(results[1])],
    [n_input * 4, 1, 1, results[2], math.exp(results[2])],
    [n_input * 8, 1, 1, results[3], math.exp(results[3])],
]

for j in range(0, 4):
    result = float(0)
    n = n_input * (2 ** j)
    t = n // (q+p)  # How many times each p and q have to be executed
    rest = n % (q+p)  # How many operations have remained
    if rest > p:
        for i in range(t*p+1):
            result = result + 1/(2 * i + 1)
        rest = rest - p
        for j in range(1, t*q + rest + 1):
            result = result - 1/(2 * j)
    else:
        for i in range(t*p+rest+1):
            result = result + 1 / (2 * i + 1)
        for j in range(1, t * q + 1):
            result = result - 1 / (2 * j)
    results.append(result)
    data.append([n, p, q, result, math.exp(result)])

table = tabulate(data, headers="firstrow", tablefmt="fancy_grid")

print(table)


