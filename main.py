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
    [n_input, 0, 0, results[0], math.exp(results[0])],
    [n_input * 2, 0, 0, results[1], math.exp(results[1])],
    [n_input * 4, 0, 0, results[2], math.exp(results[2])],
    [n_input * 8, 0, 0, results[3], math.exp(results[3])],
]

for j in range(0, 4):
    result = float(0)
    n = n_input * (2 ** j)
    i = 1
    while i <= n:
        p1 = q1 = 0
        i1 = i  # We clone i so that I can start the additions and subtractions from i but also be able
        # to check whether if it has gotten bigger than n in the while
        while p1 < p and i + p1 <= n:
            d = i % 2
            result = result + 1/(i+2*p1+(1-d))
            p1 = p1 + 1
            # With this d I make sure that n+2*p is always odd so that it's positive
        i1 = i1 + p1
        while q1 < q and i1 + q1 <= n:  # We check that the number of elements in the sum
            # isn't bigger than n. We already have i+p1=i1 elements in the sum
            d = i % 2
            result = result - 1/(i+2*q1+d)
            q1 = q1 + 1
            #  With this d I make sure that n+2*q is always even so that it's negative
            if i1 + q1 + d > n:
                break
        i = i1 + q1
    results.append(result)
    data.append([n, p, q, result, math.exp(result)])

table = tabulate(data, headers="firstrow", tablefmt="fancy_grid")

print(table)


