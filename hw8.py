import random
import numpy as np
import matplotlib.pyplot as plt


def gen_random(p_norm):
    while True:
        p_norm = int(p_norm)
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if (abs(x) ** p_norm + abs(y) ** p_norm) ** (1 / p_norm) <= 1:
            return x, y


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    p_norms = [1.25, 1.5, 3, 8]
    n = int(input("How many numbers to make in the circle: "))
    opt = int(input("Which p: 1st 2nd 3rd or 4th: ")) -1
    for i in range(n):
        a, b = gen_random(p_norms[opt])
        ax.scatter(a, b)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Unit ball for p-norm = {p_norms[opt]}")
    ax.legend()
    plt.show()
